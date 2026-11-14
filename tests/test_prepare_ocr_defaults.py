from documind.data import prepare_ocr


def test_prepare_ocr_default_check_is_disabled():
    args = prepare_ocr.build_parser().parse_args([])
    assert args.check == 0
