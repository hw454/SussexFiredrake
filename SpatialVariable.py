from firedrake import *
import numpy as np
class spatial_variable:
    def __init__(s,V,r_mesh,name=''):
        s.var=Function(V)
        s.init_var=Function(V)
        s.var_hist=[]
        s.name=name
        s.mesh=r_mesh
        s.function_space=V
        s.char_key='o' # Indictates other for when the character key is not set
    def set_name(s,namestr):
        s.name=namestr
        return
    def set_char_key(s,ck):
        s.char_key=ck
        return
    def get_name(s):
        return s.name
    def get_var(s):
        return s.var
    def get_mesh(s):
        return s.mesh
    def get_function_space(s):
        return s.function_space
    def var_vals(s):
        return s.var.vector()
    def print_var_vals(s):
        return str(s.var_vals())
    def __str__(s):
        rstr=' Spatial Variable '+s.name+'\n'
        rstr+=' Function Value '+s.print_var_vals()
        return rstr
    def update_var(s,cls_2):
        s.var.assign(cls_2.var.copy(deepcopy=True))
        s.var_hist.append(cls_2.var.copy(deepcopy=True))
        return
    def reset_var(s):
        s.var=Function(s.function_space)
        return s.var
    def initialise(s,Steady,permval,hx,hy,char='n'):
        #perm and StSt must be declared as global
        global StSt,perm_e,perm_t
        StSt=Steady
        perm_e,perm_t=permval
        x, y = SpatialCoordinate(s.mesh)
        if char=='n':
            print('Character key not input so using class key',s.char_key)
            char=s.char_key
        #perm=0
        if char=='e':
            s.var.interpolate(StSt[0]+0.005*perm_e*(cos(x)+cos(y)))
        elif char=='t':
            s.var.interpolate(StSt[1]+0.005*perm_t*(cos(x)+cos(y)))
        else:
            print('No initial func for character key')
            s.var.interpolate(np.exp(-5*(x[0]**2+x[1]**2)))
        s.init_var.assign(s.var.copy(deepcopy=True))
        s.var_hist.append(s.init_var.copy(deepcopy=True))
        return s.var
        
def initial_func(x, a=5):
    return np.exp(-a*(x[0]**2+x[1]**2))