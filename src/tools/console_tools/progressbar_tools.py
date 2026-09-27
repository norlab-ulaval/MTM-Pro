# coding=utf-8
from tqdm import tqdm


def init_progressbar(
    pb_len: int, description: str, bar_format: str = "{l_bar}{bar}| {n_fmt}/{total_fmt}"
) -> tqdm:
    """
    Initializes and returns a customizable progress bar using the tqdm library.
    Its the user responsability to close the progressbar when done.

    Usage:

        >>> progressbar = init_progressbar(99, "My cool PB")
        >>> for each in range(99):
        >>>     some_computation()
        >>>     progressbar.update()
        >>> progressbar.close()

    This helper set the progressbar to disapear from the console provided nothing was print on
    it before calling `progressbar.close()`

    :param pb_len: The length of the progress bar, indicating the total number of iterations
     or tasks to be completed.
    :param description: A short descriptive text to display as the label of the progress bar.
    :param bar_format: Default to a slim progressbar format. Set to `None` to use the tqdm default.
    :return: A tqdm progress bar object configured with the specified length and description.
    """

    progressbar = tqdm(
        range(pb_len),
        desc=description,
        bar_format=bar_format,
        leave=False,
        position=0,
    )
    return progressbar
