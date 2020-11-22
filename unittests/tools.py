"""
Common tools used by the unit tests.
"""


class Spinner:
    """
    Renders a spinner on the terminal.

    Methods
    -------
    render()
        Renders the next frame of the selected spinner.
    stop()
        Renders the stop character of the spinner.

    Examples
    --------
    import time
    spinner = Spinner()
    for i in range(10):
        spinner.render()
        time.sleep(.5)
    spinner.stop()
    """

    def __init__(self, spinner_type='dots'):
        self.started = False
        self.current_frame = 0
        # Spinner data got from https://github.com/manrajgrover/py-spinners/blob/master/spinners/spinners.py
        self.spinners = \
            {
                "dots": {
                    "interval": 80,
                    "frames": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                },
            }
        self.frames = self.spinners[spinner_type]['frames']

    def _start(self):
        self.started = True
        print("  ", sep='', end='', flush=True)

    def render(self):
        if not self.started:
            self._start()
        print("\b\b{} ".format(self.frames[self.current_frame]), sep='', end='', flush=True)
        self.current_frame = (self.current_frame + 1) % len(self.frames)

    def stop(self, stop_character=''):
        self.started = False
        print("\b\b{}".format(stop_character), sep='', end='', flush=True)


if __name__ == "__main__":
    import time
    print("Testing the spinner: ", sep='', end='')
    spinner = Spinner()
    for i in range(10):
        spinner.render()
        time.sleep(.5)
    spinner.stop()
    print("Done!")
