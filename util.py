def print_debug(*args, sep=" ", end="\n", file=None, flush=False, debug=False):
    """
    Print with a debug toggle, same signature as print with the debug argument
    :param args: All args to print
    :param sep: Separator
    :param end: Line ending
    :param file: File to write to
    :param flush: Whether to clear output buffer
    :param debug: Toggle debug, True for debug mode and vice versa
    :return:
    """
    if debug:
        print(*args, sep=sep, end=end, file=file, flush=flush)