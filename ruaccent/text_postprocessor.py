def fix_capital(source, target):
    if len(source) != len(target):
        return target
            
    mask = [x.isupper() for x in source]
    return "".join([tgt.upper() if m else tgt.lower() for m, tgt in zip(mask, target)])
        


def fix_capital_stressed(source, target):
    """Carry the capitalisation of ``source`` onto ``target`` across ``+`` marks.

    ``fix_capital`` compares lengths and bails out, which the stress mark always
    triggers, so stressed variants need their own copy of the mask.
    """
    if len(target.replace("+", "")) != len(source):
        return target

    result = []
    index = 0
    for char in target:
        if char == "+":
            result.append(char)
            continue
        result.append(char.upper() if source[index].isupper() else char)
        index += 1
    return "".join(result)
