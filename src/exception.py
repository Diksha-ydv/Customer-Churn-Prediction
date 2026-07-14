import sys 

class CustomerChurnException(Exception):
    def __init__(self,error,error_detail:sys):
        self.error = error 
        _,_,exc_tb = error_detail.exc_info()
        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.line_no = exc_tb.tb_lineno

    def __str__(self):
        return 'error occured in python script name [{0}] line no [{1}] error detail [{2}]'.format(self.file_name,self.line_no,self.error)
    

