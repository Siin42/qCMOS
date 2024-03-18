import inspect



def get_def_name():
    # print(f"This message is printed from {inspect.currentframe().f_back.f_code.co_name}")
    def_name = 'DEF ' + inspect.currentframe().f_back.f_code.co_name + '()'
    return def_name