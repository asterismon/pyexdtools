import cmath
import ctypes
import functools
import math
import os
import time
import colorama
import requests
from typing import Any, Callable, Dict, List, Tuple, Union, overload  # using for pylint
from threading import Thread
from numpy import ndarray

colorama.init(autoreset=False)

__version__ = "0.4.6"

def asyncFuncTimer(func: Callable) -> Callable:
    '''
    异步测定函数运行时间(无返回值)
    误差范围0~+10ms
    '''
    def __timer(*args, **kwargs) -> None:
        nonlocal func
        start = time.perf_counter_ns()
        func_running = Thread(target=func,args=args,kwargs=kwargs)
        func_running.run()
        while func_running.isAlive():
            time.sleep(10)
            stop = time.perf_counter_ns()
            print(f"<{func.__name__}>运行中...已运行{(stop-start)/1_000_000} ms")
        print(f"<{func.__name__}>已结束,运行了{(stop-start)/1_000_000} ms")
    return __timer


def funcTimer(func: Callable) -> Callable:
    '''
    测定函数运行时间的装饰器(无返回值)
    A decorator to count time the func uses
    '''
    def __timer(*args, **kwargs) -> None:
        nonlocal func
        start = time.perf_counter_ns()
        func(*args, **kwargs)
        stop = time.perf_counter_ns()
        print(f'<{func.__name__}>耗时{stop-start}ns({float(stop-start)/1_000_000_000} seconds)')
    return __timer


def codeTimer(func: Callable) -> Callable:#NEED TEST
    '''
    测定函数运行时间并返回其运行值
    count time the func uses, and return the result
    '''
    @funcTimer
    @functools.wraps(func)
    def __codeTimer(*args, **kwargs) -> Any:
        nonlocal func
        return func(*args, **kwargs)
    return __codeTimer


class Lnum:
    '''
    受限数值(Limited Number)
    创建实例后使用 实例名称()来获取数值

    Raises:
        Lnum.RangeError
        Lnum.OutofRangeError
    '''
    class RangeError(Exception):pass
    class OutofRangeError(Exception):pass

    def __init__(self, num: Union[int, float], minimum: Union[int, float] = float('-inf'), maximum: Union[int, float] = float('inf')) -> None:
        self.num:Union[int, float] = num
        self.mini:Union[int, float] = minimum
        self.maxi:Union[int, float] = maximum

    def __setattr__(self, __name: str, __value: Any) -> None:
        self.__dict__[__name] = __value
        if hasattr(self, 'mini') and hasattr(self, 'maxi'):
            if self.mini > self.maxi:
                raise Lnum.RangeError(f'Invalid range {self.mini}~{self.maxi}')
        if hasattr(self, 'maxi'):
            limited_num_not_in_range = self.num < self.mini or self.num > self.maxi
            if limited_num_not_in_range:
                raise Lnum.OutofRangeError(f'Number({self.num}) out of range({self.mini}~{self.maxi})')

    def __str__(self) -> str:
        return f'LimitedNum({self.num} in [{self.mini}~{self.maxi}])'

    def __call__(self) -> Union[int, float]:
        return self.num


class Lstr:
    '''
    受限字符串(Limited String)
    限制字符串的长度。限制字符串内容应使用typing.Literal

    Raises:
        Lstr.RangeError
        Lstr.OutOfLengthError
    '''
    class RangeError(Exception):
        pass

    class OutOfLengthError(Exception):
        pass

    def __init__(self, string:str, min_len:float=0, max_len:float=float("inf"), no_error:bool=False) -> None:
        self.string:str = string
        self.min_len:float = min_len
        self.max_len:float = max_len
        self.no_error = no_error

    def __setattr__(self, __name, __value) -> None:
        if hasattr(self, 'string'):
            cp_str = self.string
        else:
            self.__dict__[__name] = __value
            return
        self.__dict__[__name] = __value
        lens = len(self.string)
        if hasattr(self, 'min_len') and hasattr(self, 'max_len'):
            if self.min_len > self.max_len:
                raise Lstr.RangeError(f'Invalid range {self.min_len}~{self.max_len}')
        if hasattr(self, 'no_error'):
            string_not_in_range = lens < self.min_len or lens > self.max_len
            if string_not_in_range:
                if self.no_error:
                    self.string = cp_str
                else:
                    raise Lstr.OutOfLengthError(f'StringLen({len(self.string)}) out of range({self.min_len}~{self.max_len})')

    def __str__(self) -> str:
        return f'LimitedStr({self.string}, {len(self.string)} in [{self.min_len} ~ {self.max_len}])'

    def __call__(self) -> str:
        return self.string


class Snum:
    @overload
    @staticmethod
    def sqrt(num: Union[int, float]) -> Union[int, float, complex]: ...

    @overload
    @staticmethod
    def sqrt(num: complex) -> complex: ...

    @staticmethod
    def sqrt(num):
        '''
        返回数字的平方根
        '''
        if isinstance(num, complex) or num < 0:
            rst = cmath.sqrt(num)
        elif num == 0:
            rst = 0.0
        elif num > 0:
            rst = math.sqrt(num)
        else:
            raise ValueError
        return rst


class Sstr:
    @staticmethod
    def keepSplit(_str: str, sep: str，max_split:int = -1) -> List[str]:
        '''
        字符串分隔成数组时保留分割字符
        '''
        split = []
        while len(split)+1 == max_split and len(str) != 0:
            if sep in _str:
                find = _str.find(sep)
                split.append(_str[:find])
                split.append(sep)
                _str = _str[find+len(sep):]
            else:
                split.append(_str)
                break
        else:
            split.append(_str)
        return split

    @staticmethod
    def cut(_str: str, partLen: int) -> List[str]:
        '''
        将字符串切成等长的数份
        '''
        return [_str[i:i+partLen] for i in range(0, len(_str), partLen)]


class Sbool: #UNFINISHED
    pass


class BetterFloat:  #TODO UNFINISHED
    def __init__(self,_int:int=0,_float:int=0) -> None:
        self.i:int = _int
        self.f:int = _float if self.i>0 else -_float

    @property
    def num(self) -> str:
        return f'{self.i}.{self.f if self.f>0 else -self.f}'

    def __div__(self, other: "BetterFloat") -> "BetterFloat":#TODO
        return self.__mul__(BetterFloat(str(1/other.num)))

    def __neg__(self) -> "BetterFloat":#TODO
        strF = list(self.spt)
        strF[0] = '-' if strF[0] == '+' else ''
        return BetterFloat(''.join(strF))

    def __add__(self, other: "BetterFloat") -> "BetterFloat":#TODO
        ofp = [int(x) for x in list(str(other.fpt))]
        sfp = [int(x) for x in list(str(self.fpt))]
        for _index, num in enumerate(ofp):
            try:
                sfp[_index] += num
            except:
                sfp.append(num)
        sfp.reverse()
        for _index, i in enumerate(sfp):
            if i >= 10:
                sfp[_index] -= 10
                if _index+1 != len(sfp):
                    sfp[_index+1] += 1
                else:
                    self.ipt += 1
        sfp.reverse()
        saving = ''.join([str(x) for x in sfp])
        return BetterFloat(str(self.ipt + other.ipt)+'.'+saving)

    def __sub__(self, other: "BetterFloat") -> "BetterFloat":#TODO
        if other.num >= 0:
            ofp = [int(x) for x in list(str(other.fpt))]
            sfp = [int(x) for x in list(str(self.fpt))]
        else:
            return self.__add__(-other)
        for _index, num in enumerate(ofp):
            try:
                sfp[_index] -= num
            except:
                sfp.append(-num)
        for _index, i in enumerate(sfp):
            if i < 0:
                if _index != 0:
                    sfp[_index-1] -= 1
                else:
                    self.ipt -= 1
                sfp[_index] += 10
        saving = ''.join([str(x) for x in sfp])
        return BetterFloat(str(self.ipt - other.ipt)+'.'+saving)

    def __mul__(self, other: "BetterFloat") -> "BetterFloat":#TODO
        point = len(str(self.fpt))+len(str(other.fpt))
        saving = list(str(int(f'{self.ipt}{self.fpt}')
                      * int(f'{other.ipt}{other.fpt}')))
        saving.insert(len(saving)-point, '.')
        return BetterFloat(''.join(saving))

    def __str__(self) -> str:
        return f'BetterFloat({self.i}.{self.f if self.f>0 else -self.f})'

    def __call__(self) -> str:
        return self.num


class Sfloat:
    @staticmethod
    def round(num:float,digit:int=0)->float:#TEST
        '''
        四舍五入
        '''
        num*=digit if digit!=0 else 1
        return int(num+5)/digit if digit!=0 else int(num+5)
    
    @staticmethod
    def round46(num:float,digit:int=0)->float:#TEST
        '''
        四舍六入
        '''
        num*=digit if digit!=0 else 1
        if (num-5)%10 == 0:
            return int(num)/digit if digit!=0 else int(num)
        else:
            return int(num+4)/digit if digit!=0 else int(num+4)


class SuperByte: #UNFINISHED
    pass


class Slist:
    @staticmethod
    def merge(*Lists) -> list:#TEST
        '''
        合并列表
        '''
        return sum(lists,[])
    
    @staticmethod
    def AND(*Lists) -> list:#TEST
        '''
        提取列表中所有共同元素
        '''
        return list(set(lists[0]).intersection(*list({x} for x in lists[1:])))

    @staticmethod
    def flatten(List: list) -> list:
        '''
        展平多层列表
        '''
        def __set_or_tuple_flatten(set_or_tuple: Union[set, tuple, frozenset]) -> list:
            rs = []
            for x in set_or_tuple:
                if type(x) in [set, tuple, frozenset]:
                    rs.append(__set_or_tuple_flatten(x))
                elif type(x) is dict:
                    rs.append(__flattenValue(x))
                else:
                    rs.append(x)
            return rs

        def __flattenValue(val: Any) -> Union[list, Any]:
            if type(val) in [set, tuple]:
                return __set_or_tuple_flatten(val)
            elif type(val) is not dict:
                return val
            rst = [x for x in val.values()]
            rs = []
            for v in rst:
                if type(v) is dict:
                    rs.append(__flattenValue(v))
                elif type(v) in [set, tuple]:
                    rs.append(__set_or_tuple_flatten(v))
                else:
                    rs.append(v)
            return rs

        def __flatten(value: Any) -> list:
            return sum(([__flattenValue(x)] if not isinstance(x, list) else __flatten(x) for x in value), [])
        return __flatten(List)


class SuperDict:
    @staticmethod
    def merge(*Dicts:dict,**kwargs) -> dict:#TEST
        '''
        合并字典(对冲突的key,保存最右侧传入的参数)
        '''
        return {**{key:value for Dict in dicts for key,value in Dict.items()},**kwargs}

    @staticmethod
    def AND(*Dicts:dict) -> list:
        '''
        提取key值相同的键值对
        '''
        return [(key,value) for key, value in SuperDict.merge(*Dicts).items() if key in Slist.AND(*Dicts)]

    @staticmethod
    def __set_or_tuple_flatten(set_or_tuple: Union[set, tuple, frozenset]) -> list:
        rs = []
        for x in set_or_tuple:
            if type(x) in [set, tuple, frozenset]:
                rs.append(Stuple.flatten(x))
            elif type(x) is dict:
                rs.append(SuperDict.flattenValue(x))
            else:
                rs.append(x)
        return Slist.flatten(rs)

    @staticmethod
    def flattenValue(Dict: dict) -> list:
        '''
        展平value
        '''
        rst = [x for x in Dict.values()]
        rs = []
        for v in rst:
            if type(v) is dict:
                rs.append(SuperDict.flattenValue(v))
            elif type(v) in [set, tuple]:
                rs.append(SuperDict.__set_or_tuple_flatten(v))
            else:
                rs.append(v)
        return Slist.flatten(rs)

    @staticmethod
    def flattenKey(Dict: dict) -> list:
        '''
        展平key
        '''
        rst = [x for x in Dict.keys()]
        rs = []
        for v in rst:
            if type(v) is dict:
                rs.append(SuperDict.flattenValue(v))
            elif type(v) in [set, tuple]:
                rs.append(SuperDict.__set_or_tuple_flatten(v))
            else:
                rs.append(v)
        return Slist.flatten(rs)

    @staticmethod
    def flatten(Dict: dict) -> dict:
        '''
        展平多层字典
        '''
        return {key: value for key, value in zip(SuperDict.flattenKey(Dict), SuperDict.flattenValue(Dict))}

class Stuple:
    @staticmethod
    def merge(*Tuples: tuple) -> tuple:#TEST
        '''
        合并元组
        '''
        return tuple({element for element in Tuple for Tuple in Tuples})

    @staticmethod
    def AND(*Tuples) -> tuple:#TEST
        '''
        取所有元组中的共同元素(无序)
        '''
        return tuple([val[1] for val in {(index,element) for index,element in enumerate(Tuple) for Tuple in Tuples}])

    @staticmethod
    def flatten(Tuple: tuple, depth: int = 0) -> Union[tuple, list]:
        rs = []
        for x in Tuple:
            if type(x) in [set, tuple, frozenset]:
                rs.append(Stuple.flatten(x, depth+1))
            elif type(x) is dict:
                rs.append(SuperDict.flattenValue(x))
            else:
                rs.append(x)
        if depth == 0:
            return tuple(Slist.flatten(rs))
        return rs


class Sset:
    @staticmethod
    def AND(*Sets:set) -> set:
        return Sets[0].union(*Sets[1:])

    @staticmethod
    def OR(*Sets: set) -> set:
        return Sets[0].intersection(*Sets[1:])

    @staticmethod
    def flatten(Set: set, depth: int = 0) -> Union[set, list]:
        rs = []
        for x in Set:
            if type(x) in [set, tuple, frozenset]:
                rs.append(Sset.flatten(x, depth+1))
            elif type(x) is dict:
                rs.append(SuperDict.flattenValue(x))
            else:
                rs.append(x)
        if depth == 0:
            return set(Slist.flatten(rs))
        return rs


class SListNode: #TODO TEST
    try:
        from datanode import ListNode
    except:
        print(colorama.Fore.LIGHTYELLOW_EX +
              "WARNING:class <SListNode> is unavailable,because module 'datanode' is not installed" + colorama.Fore.RESET)

    @staticmethod
    def AND(*Nodes: "SListNode.ListNode") -> "SListNode.ListNode":
        newNodes = [node.tolist() for node in Nodes]
        return SListNode.ListNode.createByList(Slist.merge(*newNodes))
    @staticmethod
    def OR(*Nodes: "SListNode.ListNode") -> "SListNode.ListNode":
        newNodes = [node.tolist() for node in Nodes]
        return SListNode.ListNode.createByList(Slist.OR(*newNodes))
    

class SuperObj:
    __color: Dict[Union[str, type, None, str], str] = {
        'obj': colorama.Fore.LIGHTGREEN_EX,
        'var': colorama.Fore.LIGHTRED_EX,
        str: colorama.Fore.YELLOW,
        int: colorama.Fore.LIGHTMAGENTA_EX,
        float: colorama.Fore.LIGHTMAGENTA_EX,
        None: colorama.Fore.BLUE,
        list: colorama.Fore.LIGHTCYAN_EX,
        dict: colorama.Fore.LIGHTCYAN_EX,
        set: colorama.Fore.LIGHTCYAN_EX,
        tuple: colorama.Fore.LIGHTCYAN_EX,
        bool: colorama.Fore.BLUE,
        'RESET': colorama.Fore.RESET,
        'SYMBOL': colorama.Fore.LIGHTWHITE_EX
    }

    @staticmethod
    # 从ID获取值
    def __unpackClass(objID: int, res: list, spaces: int, depth: int = 0, test=False, max_depth: Union[int, float] = float('inf')) -> list:
        Object = ctypes.cast(objID, ctypes.py_object).value
        if (depth := depth + 1) > max_depth:
            res.append(' '*spaces*depth+colorama.Fore.LIGHTRED_EX +
                       '...'+colorama.Fore.RESET+'\n')
            return []
        elif type(Object) is dict:
            return SuperObj.__unpackDict(Object, res, spaces, depth, max_depth)
        elif hasattr(Object, '__dict__'):
            return SuperObj.__unpackDict(Object.__dict__, res, spaces, depth, max_depth)
        else:
            return []

    @staticmethod
    # 拆解dict
    def __unpackDict(dicts: dict, res: list, spaces: int, depth: int, max_depth: Union[int, float]):
        if depth > max_depth:
            return res
        for key, value in dicts.items():
            res.append(' '*spaces*depth)
            res.append('LATTER_IS_VARIABLE_TAG')  # 标记变量
            res.append(str(key))
            res.append(':')
            if value is not None and isinstance(value,type(value)) and type(value) not in [int,float,str,bool]:
                res.append('LATTER_IS_ID_TAG2')  # 标记类ID
                res.append(str(type(value))[8:-2]+', ID:')
                res.append(id(value))
                if type(value) in [list, dict, set, tuple]:
                    val = str(value)
                    while ' object at 0x' in val:
                        ot = val[val.find(' object at 0x')+11:]
                        rt = val[:val.find(' object at 0x')]+', ID:'
                        num, ot = ot[:ot.find('>')], ot[ot.find('>'):]
                        val = rt+str(int(num, 16))+ot
                    res.append('\n'+' '*spaces*(depth)+' '*(len(key)+1) +
                               SuperObj.__color[type(value)]+str(val)+SuperObj.__color['RESET'] +
                               '\n'+' '*spaces*(depth)+' '*(len(key)+1) +
                               SuperObj.__color['obj'] +
                               '<'+str(type(value))[8:-2]+', ID:'+str(id(value))+'>:' +
                               SuperObj.__color['RESET'] +
                               '\n'
                               )
                else:
                    res.append('\n')
                if id(value) not in res[:-2]:
                    if type(value) in [list, dict, set, tuple]:
                        value = list(SuperList.flatten([value]))
                        try:
                            value.sort()
                        except:
                            pass
                        for i in value:
                            try:
                                SuperObj.__unpackClass(
                                    id(i), res, spaces, depth, max_depth=max_depth)
                            except:
                                res.append(colorama.Fore.LIGHTRED_EX +
                                           colorama.Back.LIGHTYELLOW_EX +
                                           'Error:Unpack Failed' +
                                           colorama.Fore.RESET +
                                           colorama.Back.RESET)
                    else:
                        try:
                            if type(value) is str:
                                raise TypeError
                            SuperObj.__unpackClass(
                                id(value), res, spaces, depth, max_depth=max_depth)
                        except:
                            if type(value) in SuperObj.__color:
                                res.append(' '*spaces*(depth)+' '*(len(key)+1) +SuperObj.__color[type(value)]+str(value))
                                res.append('\n')
                            elif value is None:
                                res.append(' '*spaces*(depth)+' '*(len(key)+1) +SuperObj.__color[None]+"None")
                                res.append('\n')
                            else:
                                res.append(' '*spaces*(depth)+' '*(len(key)+1) +value)
                                res.append('\n')
                else:
                    res[-4] = 'AFTER_HAS_EXIST_ID_TAG'
            else:
                res.append(value)
                res.append('\n')
        return res

    @staticmethod
    def __unpackDictVal(_index: list, dict_val: dict, res: list, spaces: int, depth: int, max_depth: Union[int, float]):
        if (depth := depth + 1) > max_depth:
            res.insert(_index[0], '...')
            return res
        for key, value in dict_val.items():
            res.insert(_index[0], ' '*spaces*depth)
            _index[0] += 1
            res.insert(_index[0], 'LATTER_IS_VARIABLE_TAG')
            _index[0] += 1
            res.insert(_index[0], key)
            _index[0] += 1
            res.insert(_index[0], ':')
            _index[0] += 1
            if type(value) is dict:
                res.insert(_index[0], '{\n')
                _index[0] += 1
                SuperObj.__unpackDictVal(
                    _index, value, res, spaces, depth, max_depth)
                res.insert(_index[0], ' '*spaces*depth+'}\n')
                _index[0] += 1
            else:
                res.insert(_index[0], value)
                _index[0] += 1
                res.insert(_index[0], '\n')
                _index[0] += 1
        return res

    @staticmethod
    def __unpackValue(list_val: list, spaces: int, depth: int, max_depth: Union[int, float]) -> str:
        if (depth := depth+1) > max_depth:
            return "..."
        res = list_val
        while True:
            dictUPK: list[Any] = []
            for num, i in enumerate(res):
                if type(i) is dict:
                    res[num] = '\n'
                    SuperObj.__unpackDictVal(
                        [num+1], i, res, spaces, depth, max_depth)
            if dictUPK == []:
                break
        while True:
            listUPK: list[Any] = []
            for num, i in enumerate(res):
                if type(i) is list:
                    res[num] = '\n'
                    [str(x)+'\n' for x in res[num]]
            if listUPK == []:
                break
        return ''.join([str(x) for x in res]).replace('LATTER_IS_VARIABLE_TAG', '')

    @staticmethod
    def __resultDealing(rst: list, spaces: int, max_depth: Union[int, float], _color: dict = __color):
        jump = 0
        rest_jumps = 0
        rst2 = rst.copy()
        for num, i in enumerate(rst2):
            if jump:
                jump -= 1
                continue
            if i == 'LATTER_IS_ID_TAG2':
                rst[num+rest_jumps] = _color['obj']
                rst.insert(num+rest_jumps+1, '<')
                rst[num+rest_jumps+3] = str(rst[num+rest_jumps+3])
                rst.insert(num+rest_jumps+4, '>')
                rst.insert(num+rest_jumps+5, _color['RESET'])
                jump = 2
                rest_jumps += 3
            elif str(i).endswith('LATTER_IS_ID_TAG'):
                i = i[:i.find('LATTER_IS_ID_TAG')]
                rst[num+rest_jumps] = _color['obj']
                rst.insert(num+rest_jumps+1, i+'<')
                rst[num+rest_jumps+3] = str(rst[num+rest_jumps+3])
                rst.insert(num+rest_jumps+4, '>:')
                rst.insert(num+rest_jumps+5, _color['RESET'])
                jump = 2
                rest_jumps += 3
            elif i == 'AFTER_HAS_EXIST_ID_TAG':
                rst[num+rest_jumps] = _color['obj']
                rst.insert(num+rest_jumps+1, '<')
                rst[num+rest_jumps+3] = str(rst[num+rest_jumps+3])
                rst.insert(num+rest_jumps+4, '>:')
                rst.insert(num+rest_jumps+5, _color['SYMBOL'])
                rst.insert(num+rest_jumps+6, '...')
                rst.insert(num+rest_jumps+7, _color['RESET'])
                jump = 2
                rest_jumps += 5
            elif i == 'LATTER_IS_VARIABLE_TAG':
                rst[num+rest_jumps] = _color['var']
                rst.insert(num+rest_jumps+2, _color['RESET'])
                jump = 1
                rest_jumps += 1
            else:
                if i == '\n' or rst2[num-1] == '\n':
                    continue
                elif type(i) == dict:
                    rst.insert(num+rest_jumps, _color[dict])
                    rst.insert(num+rest_jumps+1, ' '*len(rst2[num-8])+'{')
                    rst[num+rest_jumps+2] = SuperObj.__unpackValue(
                        list_val=[i],
                        spaces=spaces,
                        depth=int(len(rst2[num-8])/4-1),
                        max_depth=max_depth
                    )[:-1]+'\n'+' '*len(rst2[num-8])+'}'
                    rest_jumps += 2
                else:
                    try:
                        if rst[num+rest_jumps] is not None:
                            if i == ':':
                                rst.insert(num+rest_jumps, _color['SYMBOL'])
                            else:
                                rst.insert(num+rest_jumps, _color[type(i)])
                        else:
                            rst.insert(num+rest_jumps, _color[None])
                    except:
                        rst.insert(num+rest_jumps, _color[str])
                    if str(rst[num+rest_jumps-2]) == ':':
                        if not isinstance(rst[num+rest_jumps+1], str):
                            rst[num+rest_jumps+1] = str(rst[num+rest_jumps+1])
                        else:
                            rst[num+rest_jumps +
                                1] = f'"{str(rst[num+rest_jumps+1])}"'
                    else:
                        rst[num+rest_jumps+1] = str(rst[num+rest_jumps+1])
                    rst.insert(num+rest_jumps+2, _color['RESET'])
                    rest_jumps += 2
        return rst

    @staticmethod
    def objlist(obj, depth=float('inf'), spaces=4, save=False):
        result = SuperObj.__resultDealing(
            SuperObj.__unpackClass(objID=id(obj),
                                                                 res=['LATTER_IS_ID_TAG',
                                                                      str(type(obj))[
                                                                          8:-2]+', ID:',
                                                                      id(obj),
                                                                      '\n'
                                                                      ],
                                                                 spaces=spaces,
                                                                 max_depth=depth
                                                                 ),
                                          spaces=spaces,
                                          max_depth=depth
                                          )
        if save:
            result = ''.join(result)
            for i in SuperObj.__color.values():
                result = result.replace(i, '')
            with open(f'./{id(obj)}.txt', 'w') as f:
                f.write(result)
        return [str(x) for x in result]

    @staticmethod
    def objprint(obj, depth=float('inf'), spaces=4, notes=True):
        print(''.join(SuperObj.objlist(obj, depth, spaces)))
        if notes:
            for num, i in enumerate(SuperObj.__color):
                if isinstance(i, str):
                    k = str(i)
                elif i is None:
                    k = 'None'
                else:
                    k = str(i)[8:-2]
                if num % 2 == 0:
                    print(SuperObj.__color[i] +
                          f'■ :{k}'+' '*(10-len(k)), end='')
                else:
                    print(SuperObj.__color[i]+f'■ :{k}')
        print(SuperObj.__color['RESET'])
        return

#S(string)V(voice)I(image)
class SVI:  
    try:
        import cnocr
    except:
        print(colorama.Fore.LIGHTYELLOW_EX +
              "WARNING:class <SVI> is unavailable,because module 'cnocr' is not installed" + colorama.Fore.RESET)
    try:
        import urllib.request
    except:
        print(colorama.Fore.LIGHTYELLOW_EX +
              "WARING:class <SVI> is unavailable,because module 'urllib' is not installed" + colorama.Fore.RESET)
    try:
        from PIL import Image, ImageDraw, ImageFont
    except:
        print(colorama.Fore.LIGHTYELLOW_EX +
              "WARNING:class <SVI> is not available,because module 'pillow' is not installed" + colorama.Fore.RESET)

    @staticmethod
    def voice2str(): raise Exception('Unavailable')

    @staticmethod
    def str2voice(text, sound_ray='man', speak_speed=1, path='./', filename='str2voice'):
        sound_ray = str(sound_ray).lower()
        if sound_ray == 'man':
            sound_ray = 1
        elif sound_ray == 'shota':
            sound_ray = 0
        elif sound_ray == 'woman':
            sound_ray = 6
        elif sound_ray == 'loli':
            sound_ray = 2
        else:
            raise ValueError('INCORRECT SOUNDRAY INPUT')
        try:
            speak_speed = round(1.0/speak_speed, 1)
        except:
            raise ValueError('INCORRECT SPEAK SPEED INPUT')
        result = requests.get('https://fanyi.sogou.com/reventondc/synthesis?text={}&speed={}&lang=zh-CN&from=translateweb&speaker={}'.format(
            SVI.urllib.request.quote(text), str(speak_speed), sound_ray)).content  # type:ignore[attr-defined]
        try:
            os.makedirs(path)
        except:
            pass
        finally:
            with open('{}{}.mp3'.format(path, filename), 'wb') as f:
                f.write(result)
        return path+filename+'.mp3'

    class Path:
        "File Path"
    class Tensor:
        "torch.Tensor"

    @staticmethod
    def img2str(img: Union[str, "Path", "Tensor", ndarray], single_line=True, ReturnDetails=False, confidence_warning: Dict[str, float] = {"NoWordWarning": 0.2, "NotTrustableWarning": 0.5, "LowConfidenceWarning": 0.85}, save=False, path='./', name='img2str'):
        '''
        Args:
            img: source file
            single_line: Remove all line feeds or keep the lines. Defaults to True.
            ReturnDetails: Return more useful information like confidences and confidence-warning. Defaults to False.
            confidence_warning: Are the words trustable? Defaults to {"NoWordWarning":0.2,"NotTrustableWarning":0.5,"LowConfidenceWarning":0.85}.
            save: Save your file or not. Defaults to False.
            path: Available when 'save' is True.Depends where to save your file. Defaults to './'.
            name: Available when 'save' is True.Depends your file's name. Defaults to 'img2str'.
        '''
        file = path+name+'.txt'
        _cnocr = SVI.cnocr.CnOcr()
        result = _cnocr.ocr(img_fp=img) 
        as_is_text = [''.join(x[0]) for x in result]
        if single_line:
            text = ''.join(as_is_text)
        else:
            text = '\n'.join(as_is_text)
        if ReturnDetails:
            confidence_level = [round(x[1], 2) for x in result]
            average_confidence_level = 0.0

            def _add(inputs):
                nonlocal average_confidence_level
                average_confidence_level += inputs
            _ = [_add(x) for x in confidence_level]
            average_confidence_level /= len(confidence_level)
            if average_confidence_level <= confidence_warning['NoWordWarning']:
                warning_level = 'NoWordWarning'
                as_is_text = []
                text = ''
            elif average_confidence_level <= confidence_warning['NotTrustableWarning']:
                warning_level = 'NotTrustableWarning'
            elif average_confidence_level <= confidence_warning['LowConfidenceWarning']:
                warning_level = 'LowConfidenceWarning'
            else:
                warning_level = ""
            details = {'as_is_text': as_is_text, 'text': text, 'confidence': round(
                average_confidence_level, 2), 'all_confidence': confidence_level, 'warning': warning_level}
            if save:
                with open(file, 'w') as f:
                    f.write(str(details))
            return details
        if save:
            with open(file, 'w') as f:
                f.write(str(text))
        return text

    @staticmethod
    def str2img(text, path='./', name='str2img', find='', x=1, y=1):
        file = path+name+".png"
        if find != '':
            if x == 1:
                x = max([len(x) for x in text.split(find)])*13
            if y == 1:
                y = (text.count(find)+1)*20
        im = SVI.Image.new("RGB", (x, y), (255, 255, 255))
        dr = SVI.ImageDraw.Draw(im)
        font = SVI.ImageFont.truetype(os.path.join("./fonts", "msyh.ttf"), 14)
        dr.text((10, 5), text, font=font, fill="#000000")
        im.save(file)
        return file


class SuperType: #TO_REMOVE
    @staticmethod
    def toStr(text, find='', change_ENTER=True, keep_tuple=False, keep_set=False, dict_split=False):
        if type(text) == list:
            text = [SuperType.toStr(x, find, change_ENTER, keep_tuple, dict_split)
                    for x in text]
            if change_ENTER == True:
                text = ", ".join(text).replace(", ", "\n").replace(", ", ', ')
            else:
                text = ', '.join(text).replace(
                    ' , ', ', ').replace(", ", ', ')
        elif type(text) == dict:
            if dict_split == False:
                text = [SuperType.toStr(x, find, change_ENTER, keep_tuple=True)
                        for x in text.items()]
            else:
                text = [SuperType.toStr(x, find, change_ENTER, keep_tuple=False)
                        for x in text.items()]
            text = ", ".join(text).replace(
                ", ", " : ").replace("(", "").replace(")", "")
        elif type(text) == tuple:
            if keep_tuple == True:
                text = "("+', '.join([SuperType.toStr(x) for x in text])+")"
            else:
                text = " , ".join([SuperType.toStr(x) for x in text])
        elif type(text) in [int, float, str, bool] or text == None:
            text = str(text)
        elif type(text) == type:
            text = str(text).replace("'", '')
        elif type(text) == set:
            if keep_set == False:
                text = ' , '.join(
                    [SuperType.toStr(x, find, change_ENTER, keep_tuple, dict_split) for x in list(text)])
            else:
                text = '{'+', '.join([SuperType.toStr(x) for x in text])+'}'
        else:
            raise ValueError
        return text


class Sbit:  #TEST
    def __init__(self,val:Union[int,str] = "00000000"):
        if type(val) is int:
            __temp = []
            while val!=0:
                __temp.append(str(val%2))
                val=val//2
            val = ''.join(__temp[::-1])
        self.val = val

    def __check(self, val: Union[int, bool]):
        if type(val) is bool:
            if val:
                return 1
            else:
                return 0
        if val % 2:
            return 0
        else:
            return 1

    def _1(self, val: Union[int, bool]):
        val = self.__check(val)
        self.val = self.val[:8]+str(val)

    def _2(self, val: Union[int, bool]):
        val = self.__check(val)
        self.val = self.val[:7]+str(val)+self.val[:-1]

    def _3(self, val: Union[int, bool]):
        val = self.__check(val)
        self.val = self.val[:6]+str(val)+self.val[:-2]

    def _4(self, val: Union[int, bool]):
        val = self.__check(val)
        self.val = self.val[:5]+str(val)+self.val[:-3]

    def _5(self, val: Union[int, bool]):
        val = self.__check(val)
        self.val = self.val[:4]+str(val)+self.val[:-4]

    def _6(self, val: Union[int, bool]):
        val = self.__check(val)
        self.val = self.val[:3]+str(val)+self.val[:-5]

    def _7(self, val: Union[int, bool]):
        val = self.__check(val)
        self.val = self.val[:2]+str(val)+self.val[:-6]

    def _8(self, val: Union[int, bool]):
        val = self.__check(val)
        self.val = str(val)+self.val[:-7]

    def __call__(self):
        return int(self.val, 2)

    def __str__(self):
        return self.val