# coding=utf-8
import sys

from tools.console_tools.format import ConsoleFormat
from dataclasses import asdict
from tools.console_tools.message import consol_msg_universal_one_liner

class ConsoleRecorder(object):
    """Record to file and print to console the stdout

    Inspired by: https://stackoverflow.com/a/14906787
    """

    def __init__(self, print_to_console: bool = True):
        self.print_to_console = print_to_console
        self.terminal = sys.stdout
        self.log = open("console.log", "a")
        self.console_format = ConsoleFormat()


    def write(self, message: str) -> None:
        if self.print_to_console:
            self.terminal.write(message)

        self.log.write(self.console_format_parser(message))
        return None

    def console_format_parser(self, message: str) -> str:
        """Strip console formating character from message

        :param message: a stdout line
        :return: the stripped version
        """
        for each_key, each_format in asdict(self.console_format).items():
            message = message.replace(each_format, "")
        return message

    def flush(self) -> None:
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

    def close(self):
        msg = consol_msg_universal_one_liner(msg="Closing ConsoleRecorder", print_it=False)
        print(self.console_format_parser(msg), flush=True)
        self.log.close()
        sys.stdout = self.terminal
