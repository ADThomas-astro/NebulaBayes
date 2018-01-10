# Compatibility
if type(u"") != type(""):  # If python 2
    _str_type = basestring  # Superclass of both "str" and "unicode"
else:  # If python 3
    _str_type = str