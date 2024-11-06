"""Utilities for file io and generic data prep."""

from copy import deepcopy

from numpy import *
from scipy.interpolate import interp1d
from utils import print_verb


def get_percentile(orig_vals, percentile):
    """
    Find the percentile of a list of values.

    @parameter the_vals - is a list of values. Note the_vals MUST BE already sorted.
    @parameter percentile - a float value from 0.0 to 1.0.

    @return - the percentile of the values
    """

    the_vals = deepcopy(orig_vals)
    the_vals = sort(the_vals)

    k = (len(the_vals) - 1) * percentile
    f = floor(k)
    c = ceil(k)
    if f == c:
        return the_vals[int(k)]
    d0 = the_vals[int(f)] * (c - k)
    d1 = the_vals[int(c)] * (k - f)
    return d0 + d1


def readcol(
    file, types, splitchar=None, verbose=False
):  # types = a = one word,s = string,i = integer,f = float
    """readcol.

    Parameters
    ----------
    file :
        file path
    types :
        list with 's', 'f', 'i', 'a' for string, float, int, or any
    splitchar :
        delimiter for columns
    verbose :
        verbose
    """

    types = types.replace(",", "")

    lists = []
    for i in range(len(types)):
        lists.append([])

    try:
        fread = open(file)
        lines = fread.read()
        fread.close()
    except:
        print_verb(verbose, "Couldn't open file!")
        print_verb(verbose, file)
        return lists
    lines = lines.split("\n")

    lines = clean_lines(lines)

    linecount = 0

    if types != "s":
        for line in lines:
            parsed = line.split(splitchar)
            parsed = [item.strip() for item in parsed]

            good_line = 1
            good_entry = 0

            try:
                for i in range(len(types)):
                    if types[i] == "f":
                        float(parsed[i])
                    elif types[i] == "i":
                        int(parsed[i])
                    else:
                        parsed[i]
                    good_entry += 1
            except:
                good_line = 0
                print_verb(verbose, ("Skipping ", line, len(parsed), good_entry))

            if good_line:
                for i in range(len(types)):
                    if types[i] == "a":
                        lists[i].append(parsed[i])
                    elif types[i] == "f":
                        lists[i].append(float(parsed[i]))
                    elif types[i] == "i":
                        lists[i].append(int(parsed[i]))
                    else:
                        assert 0, "Unknown type!"
                linecount += 1

    else:
        return [lines]

    print_verb(verbose, (linecount, " lines read of ", len(lines)))

    for i in range(len(types)):

        if types[i] == "i":
            try:
                lists[i] = array(lists[i], dtype=int32)
            except:
                print_verb(verbose, ("Couldn't make array!", lists[i], file))
        if types[i] == "f":
            try:
                lists[i] = array(lists[i], dtype=float64)
            except:
                print_verb(verbose, ("Couldn't make array!", lists[i], file))

    return lists


def writecol(outputfl, data_list, headings=None, doformat=True):
    """writecol.

    Parameters
    ----------
    outputfl :
        outputfl
    data_list :
        data_list
    headings :
        headings
    doformat :
        doformat
    """
    f = open(outputfl, "w")

    if headings != None:
        f.write("#" + "  ".join(headings) + "\n")

    for i in range(len(data_list[0])):
        towrite = [str(item[i]) for item in data_list]
        f.write("  ".join(towrite) + "\n" * (i < len(data_list[0]) - 1))
    f.close()

    if doformat:
        format_file(outputfl)


def file_to_fn(fl, kind="linear", fill_value=0.0, bounds_error=True):
    """file_to_fn.
    Read a file with two columns of floats and return a fn that can interpolate x->y.

    Parameters
    ----------
    fl :
        file path
    kind :
        kind of interpolation
    fill_value :
        fill_value for interpolation
    bounds_error :
        bounds_error for interpolation
    """

    [x, y] = readcol(fl, "ff")
    return interp1d(x, y, kind=kind, bounds_error=bounds_error, fill_value=fill_value)


def read_by_space(flnm, exampleline, formats, missingval="99999", verbose=False):
    formats = formats.replace(",", "")

    begins = []
    lengths = []

    active = 0
    for i in range(len(exampleline)):
        if exampleline[i] == " ":
            # A space
            if active == 1:
                active = 0
                lengths.append(i - begins[-1])
        else:
            # Not a space
            if active == 0:
                active = 1
                begins.append(i)
    if active:
        lengths.append(len(exampleline) - begins[-1])

    if (len(lengths) != len(begins)) or (len(formats) != len(begins)):
        print_verb(verbose, "Error Parsing!")
        print_verb(verbose, exampleline)
        sys.exit(1)

    fl = open(flnm)
    lines = fl.read().split("\n")
    fl.close()

    outputs = [[] for format in formats]
    for line in lines:
        for i in range(len(begins)):
            txtclip = line[begins[i] : begins[i] + lengths[i]]
            if formats[i] == "f":
                try:
                    float(txtclip)
                    outputs[i].append(float(txtclip))
                except:
                    outputs[i].append(float(missingval))
            if formats[i] == "i":
                try:
                    int(txtclip)
                    outputs[i].append(int(txtclip))
                except:
                    outputs[i].append(int(missingval))
            if formats[i] == "a":
                if txtclip == (" " * len(txtclip)):
                    outputs[i].append(str(missingval))
                else:
                    outputs[i].append(txtclip)
    for i in range(len(formats)):
        if formats[i] == "f":
            outputs[i] = array(outputs[i], dtype=float64)
        if formats[i] == "i":
            outputs[i] = array(outputs[i])
        if formats[i] == "a":
            outputs[i] = array(outputs[i])
    return outputs


def clean_lines(orig_lines, stringlist=[""]):
    lines = deepcopy(orig_lines)
    # Start by getting rid of spaces.

    lines = [item.strip() for item in lines]

    # Check for strings to exclude.
    lines = [item for item in lines if stringlist.count(item) == 0]

    # Get rid of comments
    lines = [item for item in lines if item[0] != "#"]
    return lines


def open_lines(fl):
    f = open(fl)
    lines = f.read().split("\n")
    f.close()

    lines = clean_lines(lines)
    return lines


def file_to_dict(thefl, verbose=False):
    params = {}

    f = open(thefl)
    lines = f.read()
    lines = lines.split("\n")
    f.close()

    lines = clean_lines(lines)

    for line in lines:
        parsed = line.split(None)
        params[parsed[0]] = eval(" ".join(parsed[1:]))

    print_verb(verbose, params)
    return params


def format_file(thefile, min_space=4, max_columns=50):
    with open(thefile, "r") as f:
        lines = f.read().split("\n")

    maxwidths = [-1 for i in range(max_columns)]
    for i in range(len(lines)):
        parsed = lines[i].split(None)
        for j in range(len(parsed)):
            maxwidths[j] = max(maxwidths[j], len(parsed[j]))
    with open(thefile, "w") as f:
        for i in range(len(lines)):
            parsed = lines[i].split(None)

            if len(parsed) > 0:
                thisline = ""
                for j in range(len(parsed)):
                    thisline += parsed[j] + " " * (
                        min_space + maxwidths[j] - len(parsed[j])
                    )
                f.write(thisline + "\n" * (i < len(lines) - 1))


def get_param(flnm, param_name, default=None, verbose=False):
    # This one reads a parameter and an error from a SALT result file.

    fl = open(flnm)
    lines = fl.read().split("\n")
    lines = clean_lines(lines)
    fl.close()

    for line in lines:
        parsed = line.split(None)
        if parsed[0] == param_name:
            param_value = parsed[1:]

            if len(param_value) == 1:
                param_value.append("-1")

            param_value = [
                item * (item != "F") + "-1" * (item == "F") for item in param_value
            ]
            param_value = [eval(item) for item in param_value]

            return param_value
    print_verb(verbose, ("Couldn't get ", param_name, flnm, "returning ", default))
    return default


def read_param(flnm, param, default=None, ind=1, verbose=False):
    # This one is commonly used for photometry parameter files.

    fp = open(flnm, "r")
    lines = fp.read()
    fp.close()

    lines = lines.split("\n")
    lines = clean_lines(lines)

    for line in lines:
        parsed = line.split(None)
        if parsed[0] == param:
            print_verb(verbose, ("Reading " + param + " from " + flnm))

            try:
                return eval(parsed[ind])
            except:
                try:
                    return parsed[ind]
                except:
                    print_verb(verbose, "Found ", param, " but couldn't get ind ", ind)
                    return default
    print_verb(verbose,)
    print_verb(verbose, ("Couldn't find ", param))
    print_verb(verbose, ("Returning default ", default))
    print_verb(verbose,)

    return default


def write_param(flnm, param, value, ind=1):
    fp = open(flnm, "r")
    lines = fp.read()
    fp.close()

    lines = lines.split("\n")
    lines = clean_lines(lines)

    found = 0
    for i in range(len(lines)):
        parsed = lines[i].split(None)
        if parsed[0] == param:
            found = 1
            parsed[ind] = str(value)
            lines[i] = "  ".join(parsed)

    assert found == 1
    f = open(flnm, "w")
    f.write("\n".join(lines))
    f.close()


def get_param_nospaces(
    flnm, param_name, default=None, verbose=False
):  # This version reads everything up to comments (#), ignoring spaces
    fl = open(flnm)
    lines = fl.read().split("\n")
    lines = clean_lines(lines)
    fl.close()

    for line in lines:
        parsed = line.split(None)
        if parsed[0] == param_name:
            param_value = ""

            ind = 1

            while ind < len(parsed) and parsed[ind][0] != "#":
                # python does first condition first
                param_value += parsed[ind]
                ind += 1

            print_verb(verbose, ("Returning ", param_name, param_value))
            return eval(param_value)
    print_verb(verbose, ("Couldn't get ", param_name, " returning default ", default))
    return default


def get_list_comparison(list1, list2):
    two_but_not_one = deepcopy(list2)
    one_but_not_two = deepcopy(list1)

    try:
        one_but_not_two = one_but_not_two.tolist()
    except:
        print_verb(verbose, "list1 is a list!")

    try:
        two_but_not_one = two_but_not_one.tolist()
    except:
        print_verb(verbose, "list2 is a list!")

    for item in list1:
        while two_but_not_one.count(item) > 0:
            del two_but_not_one[two_but_not_one.index(item)]

    for item in list2:
        while one_but_not_two.count(item) > 0:
            del one_but_not_two[one_but_not_two.index(item)]

    print_verb(verbose, ("two_but_not_one", two_but_not_one))
    print_verb(verbose, ("one_but_not_two", one_but_not_two))

    return [one_but_not_two, two_but_not_one]
