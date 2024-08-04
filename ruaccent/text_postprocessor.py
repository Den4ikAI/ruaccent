def fix_capital(source, target):
    if len(source) != len(target):
        return target
            
    mask = [x.isupper() for x in source]
    return "".join([tgt.upper() if m else tgt.lower() for m, tgt in zip(mask, target)])
        
