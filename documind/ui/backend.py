"""The two ways the front end can work, behind one interface: LocalBackend (the models and the account database are used inside this process) and
ApiBackend (everything goes through the FastAPI server). The pages never know which one they talk to.

Every method that needs a signed-in user takes the session `token` as first argument (None = guest: reading works, nothing is saved).
Failures are always BackendError(status, detail) with a message that can be shown to the person as it is.
"""
from __future__ import annotations

import time


class BackendError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


class LocalBackend:
    """Runs here: documind.api.accounts (the same SQLite database the API uses, so accounts and history are shared) + the DocumentService."""

    kind = "local"

    def __init__(self, service):
        self.service = service

    # ---- helpers
    @staticmethod
    def _call(fn, *args, **kwargs):
        from fastapi import HTTPException
        try:
            return fn(*args, **kwargs)
        except HTTPException as exc:
            raise BackendError(exc.status_code, str(exc.detail)) from exc

    def _user(self, token: str | None) -> dict:
        from ..api import accounts
        user = accounts.user_or_none(f"Bearer {token}") if token else None
        if not user:
            raise BackendError(401, "Please sign in")
        return user

    # ---- models
    def available(self) -> dict:
        return self.service.available()

    # ---- account
    def register(self, name: str, email: str, password: str) -> dict:
        from ..api import accounts
        return self._call(accounts.register, accounts.Register(name=name, email=email, password=password))

    def login(self, email: str, password: str) -> dict:
        from ..api import accounts
        return self._call(accounts.login, accounts.Login(email=email, password=password))

    def logout(self, token: str) -> None:
        from ..api import accounts
        self._call(accounts.logout, authorization=f"Bearer {token}", user=self._user(token))

    def me(self, token: str) -> dict:
        from ..api import accounts
        return self._call(accounts.me, self._user(token))

    def rename(self, token: str, name: str) -> dict:
        from ..api import accounts
        return self._call(accounts.rename, accounts.Rename(name=name), self._user(token))

    def change_password(self, token: str, current: str, new: str) -> None:
        from ..api import accounts
        self._call(accounts.change_password, accounts.Password(current=current, new=new), authorization=f"Bearer {token}", user=self._user(token))

    def delete_account(self, token: str, password: str) -> None:
        from ..api import accounts
        self._call(accounts.delete_account, accounts.Delete(password=password), self._user(token))

    def stats(self, token: str) -> dict:
        from ..api import accounts
        return self._call(accounts.stats, self._user(token))

    # ---- history
    def history(self, token: str) -> list[dict]:
        from ..api import accounts
        return self._call(accounts.history, self._user(token))

    def history_item(self, token: str, hid: int) -> dict:
        from ..api import accounts
        return self._call(accounts.history_item, hid, self._user(token))

    def history_image(self, token: str, hid: int) -> bytes | None:
        from ..api import accounts
        try:
            return self._call(accounts.scan_image, hid, self._user(token)).body
        except BackendError as exc:
            if exc.status == 404:
                return None
            raise

    def delete_history(self, token: str, hid: int) -> None:
        from ..api import accounts
        self._call(accounts.history_delete, hid, self._user(token))

    def dataset_zip(self, token: str) -> bytes:
        from ..api import accounts
        return self._call(accounts.dataset_export, self._user(token)).body

    # ---- reading
    def extract(self, token: str | None, data: bytes, filename: str, doc_type: str = "auto") -> dict:
        from ..api import accounts
        from ..inference.extractor import ModelNotFound
        user = self._user(token) if token else None
        t0 = time.perf_counter()
        try:
            result = self.service.extract(data, filename, doc_type)
        except ModelNotFound as exc:
            raise BackendError(503, str(exc)) from exc
        except ValueError as exc:
            raise BackendError(422, str(exc)) from exc
        accounts.save_history(user, result, int((time.perf_counter() - t0) * 1000), data)
        return result


class ApiBackend:
    """Everything through the FastAPI server. `session` is `requests` by default; any object with get/post/patch/delete working like it does is accepted (tests)."""

    kind = "api"

    def __init__(self, base_url: str, session=None, timeout: float = 300.0):
        if session is None:
            import requests
            session = requests
        self.base, self.s = base_url.rstrip("/"), session
        self.kw = {"timeout": timeout} if session.__class__.__module__.startswith("requests") or session.__class__.__name__ == "module" else {}

    def _req(self, method: str, path: str, token: str | None = None, raw: bool = False, **kw):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            r = getattr(self.s, method)(self.base + path, headers=headers, **self.kw, **kw)
        except Exception as exc:                                   # connection refused, timeout ...
            raise BackendError(0, f"Cannot reach the API at {self.base or 'the server'}: {exc}") from exc
        if r.status_code != 200:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise BackendError(r.status_code, detail if isinstance(detail, str) else str(detail))
        return r if raw else r.json()

    def available(self) -> dict:
        return self._req("get", "/health").get("models", {})

    def register(self, name: str, email: str, password: str) -> dict:
        return self._req("post", "/auth/register", json={"name": name, "email": email, "password": password})

    def login(self, email: str, password: str) -> dict:
        return self._req("post", "/auth/login", json={"email": email, "password": password})

    def logout(self, token: str) -> None:
        self._req("post", "/auth/logout", token)

    def me(self, token: str) -> dict:
        return self._req("get", "/auth/me", token)

    def rename(self, token: str, name: str) -> dict:
        return self._req("patch", "/auth/me", token, json={"name": name})

    def change_password(self, token: str, current: str, new: str) -> None:
        self._req("post", "/auth/password", token, json={"current": current, "new": new})

    def delete_account(self, token: str, password: str) -> None:
        self._req("post", "/auth/delete", token, json={"password": password})

    def stats(self, token: str) -> dict:
        return self._req("get", "/stats", token)

    def history(self, token: str) -> list[dict]:
        return self._req("get", "/history", token)

    def history_item(self, token: str, hid: int) -> dict:
        return self._req("get", f"/history/{hid}", token)

    def history_image(self, token: str, hid: int) -> bytes | None:
        try:
            return self._req("get", f"/scans/{hid}/image", token, raw=True).content
        except BackendError as exc:
            if exc.status == 404:
                return None
            raise

    def delete_history(self, token: str, hid: int) -> None:
        self._req("delete", f"/history/{hid}", token)

    def dataset_zip(self, token: str) -> bytes:
        return self._req("get", "/dataset/export", token, raw=True).content

    def extract(self, token: str | None, data: bytes, filename: str, doc_type: str = "auto") -> dict:
        return self._req("post", "/extract", token, params={"doc_type": doc_type}, files={"file": (filename, data)})
