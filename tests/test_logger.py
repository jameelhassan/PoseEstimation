from stacked_hourglass.utils.logger import Logger


def test_logger(tmpdir):
    log_path = tmpdir.join('log.txt')
    logger = Logger(log_path)
    logger.set_names(['x', 'y'])
    for x in range(4):
        y = x ** 2
        logger.append([x, y])
    logger.close()
    expected = [
        'x\ty',
        '0.000000\t0.000000',
        '1.000000\t1.000000',
        '2.000000\t4.000000',
        '3.000000\t9.000000',
    ]
    with open(log_path, 'r') as f:
        actual = [line.strip() for line in f.readlines()]
    assert actual == expected


def test_plot_to_file(tmpdir):
    log_path = tmpdir.join('log.txt')
    plot_path = tmpdir.join('log.svg')
    logger = Logger(log_path)
    logger.set_names(['test_series_a', 'test_series_b', 'test_series_c'])
    logger.plot_to_file(plot_path, ['test_series_a', 'test_series_b'])
    with open(plot_path, 'r') as f:
        text = f.read()
        assert 'test_series_a' in text
        assert 'test_series_b' in text
        assert 'test_series_c' not in text
