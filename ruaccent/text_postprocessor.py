def fix_capital(source, target):
    if not source or not source[0].isupper():
        return target
    if not target:
        return target
    if target[0] == '+':
        if len(target) == 1:
            return target
        return target[0] + target[1].upper() + target[2:]
    else:
        return target.capitalize()
