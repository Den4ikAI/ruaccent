def fix_capital(source, target):
    """Copy letter case from ``source`` onto ``target`` of the same length."""
    if len(source) != len(target):
        return target
    return "".join(t.upper() if s.isupper() else t.lower() for s, t in zip(source, target))


def transfer_stress(word, stressed):
    """Put the "+" marks of ``stressed`` into ``word``, keeping its letters and case.

    >>> transfer_stress("Замок", "зам+ок")
    'Зам+ок'
    """
    letters = stressed.replace("+", "")
    if len(letters) != len(word):
        return stressed
    out, j = [], 0
    for ch in stressed:
        if ch == "+":
            out.append("+")
        else:
            out.append(word[j])
            j += 1
    return "".join(out)
