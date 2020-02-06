"""Utility functions"""


def ask_yes_or_no(msg, default="y"):
    """Yes or no question

    Args:
        msg     : the message or question to ask the user
        default : default response

    Returns:
        True if answer is yes, False otherwise.

    """
    response = None

    if (default == "y"):
        options = "[Y/n]"
    else:
        options = "[N/y]"

    question = msg + " " + options + " "

    while response not in {"y", "n"}:
        raw_resp = input(question) or default
        response = raw_resp.lower()

        if (response not in {"y", "n"}):
            print("Please enter \"y\" or \"n\"")

    return (response == "y")
