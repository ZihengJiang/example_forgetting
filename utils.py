import sys

# Format time for printing purposes
def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s

# Checks whether a given file name matches a list of specified arguments
#
# fname: string containing file name
# args_list: list of strings containing argument names and values, i.e. [arg1, val1, arg2, val2,..]
#
# Returns 1 if filename matches the filter specified by the argument list, 0 otherwise
#
def check_filename(fname, args_list):

    # If no arguments are specified to filter by, pass filename
    if args_list is None:
        return 1

    for arg_ind in np.arange(0, len(args_list), 2):
        arg = args_list[arg_ind]
        arg_value = args_list[arg_ind + 1]

        # Check if filename matches the current arg and arg value
        if arg + '_' + arg_value + '__' not in fname:
            print('skipping file: ' + fname)
            return 0

    return 1


def flush_print(msg):
    sys.stdout.write('\r')
    sys.stdout.write(msg)
    sys.stdout.flush()
