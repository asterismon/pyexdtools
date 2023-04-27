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

__version__ = "0.6.4"

def asyncTimer(func: Callable) -> Callable:
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
            time.sleep(0.01)
            stop = time.perf_counter_ns()
            print(f"<{func.__name__}>运行中...已运行{(stop-start)/1_000_000} ms")
        print(f"<{func.__name__}>已结束,运行了{(stop-start)/1_000_000} ms")
    return __timer


def Timer(func: Callable) -> Callable:
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
    @Timer
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
            result = cmath.sqrt(num)
        elif num == 0:
            result = 0.0
        elif num > 0:
            result = math.sqrt(num)
        else:
            raise ValueError
        return result


class Sstr:
    @staticmethod
    def keepSplit(Str: str, sep: str，max_split:int = -1) -> List[str]:
        '''
        字符串分隔成数组时保留分割字符
        '''
        split = []
        while len(split)+1 == max_split and len(str) != 0:
            if sep in Str:
                find = Str.find(sep)
                split.append(Str[:find])
                split.append(sep)
                Str = Str[find+len(sep):]
            else:
                split.append(Str)
                break
        else:
            split.append(Str)
        return split

    @staticmethod
    def cut(Str: str, sub_len: int) -> List[str]:
        '''
        将字符串切成等长的数份
        '''
        return [Str[i:i+sub_len] for i in range(0, len(Str), sub_len)]


class Sbool: #UNFINISHED
    pass


class BetterFloat:  #TODO UNFINISHED
    def __init__(self,int_part:int=0,float_part:int=0) -> None:
        self.i:int = int_part
        self.f:int = float_part if self.i>0 else -float_part

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
            result = []
            for x in set_or_tuple:
                if type(x) in [set, tuple, frozenset]:
                    result.append(__set_or_tuple_flatten(x))
                elif type(x) is dict:
                    result.append(__flattenValue(x))
                else:
                    result.append(x)
            return result

        def __flattenValue(val: Any) -> Union[list, Any]:
            if type(val) in [set, tuple]:
                return __set_or_tuple_flatten(val)
            elif type(val) is not dict:
                return val
            result = [x for x in val.values()]
            result = []
            for v in result:
                if type(v) is dict:
                    result.append(__flattenValue(v))
                elif type(v) in [set, tuple]:
                    result.append(__set_or_tuple_flatten(v))
                else:
                    result.append(v)
            return result

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
        result = []
        for x in set_or_tuple:
            if type(x) in [set, tuple, frozenset]:
                result.append(Stuple.flatten(x))
            elif type(x) is dict:
                result.append(SuperDict.flattenValue(x))
            else:
                result.append(x)
        return Slist.flatten(result)

    @staticmethod
    def flattenValue(Dict: dict) -> list:
        '''
        展平value
        '''
        result = [x for x in Dict.values()]
        result = []
        for v in result:
            if type(v) is dict:
                result.append(SuperDict.flattenValue(v))
            elif type(v) in [set, tuple]:
                result.append(SuperDict.__set_or_tuple_flatten(v))
            else:
                result.append(v)
        return Slist.flatten(result)

    @staticmethod
    def flattenKey(Dict: dict) -> list:
        '''
        展平key
        '''
        result = [x for x in Dict.keys()]
        result = []
        for v in result:
            if type(v) is dict:
                result.append(SuperDict.flattenValue(v))
            elif type(v) in [set, tuple]:
                result.append(SuperDict.__set_or_tuple_flatten(v))
            else:
                result.append(v)
        return Slist.flatten(result)

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
        result = []
        for x in Tuple:
            if type(x) in [set, tuple, frozenset]:
                result.append(Stuple.flatten(x, depth+1))
            elif type(x) is dict:
                result.append(SuperDict.flattenValue(x))
            else:
                result.append(x)
        if depth == 0:
            return tuple(Slist.flatten(result))
        return result


class Sset:
    @staticmethod
    def AND(*Sets:set) -> set:
        return Sets[0].union(*Sets[1:])

    @staticmethod
    def OR(*Sets: set) -> set:
        return Sets[0].intersection(*Sets[1:])

    @staticmethod
    def flatten(Set: set, depth: int = 0) -> Union[set, list]:
        result = []
        for x in Set:
            if type(x) in [set, tuple, frozenset]:
                result.append(Sset.flatten(x, depth+1))
            elif type(x) is dict:
                result.append(SuperDict.flattenValue(x))
            else:
                result.append(x)
        if depth == 0:
            return set(Slist.flatten(result))
        return result


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
    

class SuperObj: #TODO UNFINISHED
    from rich.text import Text
    from collections import defaultdict
    __styles: Dict[Union[str, type], str] = {
        'obj': "rgb(0,255,0)",
        'var': "rgb(156,220,254)",
        str: "rgb(255,255,0)",
        int: "rgb(255,0,255)",
        float: "rgb(128,0,128)",
        type(None): "rgb(53,140,214)",
        list: "rgb(255,165,0)",
        dict: "rgb(255,64,64)",
        set: "rgb(255,211,155)",
        tuple: "rgb(30,144,255)",
        bool: "rgb(53,140,214)",
        "error": "rgb(255,0,0)",
        'SYMBOL': "rgb(255,255,255) bold",
        "max_depth": "rgb(255,255,255)"
    }

    @staticmethod
    # 从ID获取值
    def __unpackClass(objID: int, result: Text, spaces: int, depth: int = 0, test=False, max_depth: Union[int, float] = float('inf'),loop_check: Union[None,List] = None, styles:dict = __styles) -> list:
        Object = ctypes.cast(objID, ctypes.py_object).value
        if id(Object) not in loop_check:
            result.append("<" + str(type(Object))[8:-2] + ", ID:" + str(id(Object)) + ">:",styles.setdefault(type(Object),'obj'))
            result.append('\n')
            loop_check.append(objID)
        else:
            result.append("<" + str(type(Object))[8:-2] + ", ID:" + str(id(Object)) + ">...(loop)",styles.setdefault(type(Object),'obj'))
            return []
        if (depth := depth + 1) > max_depth:
            result.append(' '*spaces*depth+'------MAX DEPTH------',styles['max_depth'])
            result.append('\n')
            return []
        elif type(Object) is dict:
            return SuperObj.__unpackDict(
                Dict=Object,
                result=result,
                spaces=spaces,
                depth=depth,
                max_depth=max_depth,
                loop_check=loop_check,
                styles=styles
            )
        elif hasattr(Object, '__dict__'):
            return SuperObj.__unpackDict(
                Dict=Object.__dict__,
                result=result,
                spaces=spaces,
                depth=depth,
                max_depth=max_depth,
                loop_check=loop_check,
                styles=styles
            )
        else:
            try:
                return SuperObj.__unpackDict(
                    Dict=dict(enumerate(Object)),
                    result=result,
                    spaces=spaces,
                    depth=depth,
                    max_depth=max_depth,
                    loop_check=loop_check,
                    styles=styles
                )
            except Exception:
                result.append(' '*spaces*(depth)+str(Object)+" Error:Unpack Failed.\n",styles['error'])
                return []

    # 拆解dict
    @staticmethod
    def __unpackDict(Dict: dict, result: Text, spaces: int, depth: int, max_depth: Union[int, float],loop_check:Union[None,List] = None,styles = __styles) -> Text:
        for key, value in Dict.items():
            result.append(' '*spaces*depth)
            result.append(str(key),styles['var'])
            if hasattr(value,'__iter__'):
                SuperObj.__unpackClass(
                    objID=id(value),
                    result=result,
                    spaces=spaces,
                    depth=depth+1,
                    max_depth=max_depth,
                    loop_check=loop_check,
                    styles=styles
                )
            elif type(value) is not str:
                try:
                    SuperObj.__unpackClass(id(value), result, spaces, depth+1, max_depth=max_depth)[0]
                    return
                except Exception:
                    pass
                if type(value) in styles:
                    result.append(str(value), styles[type(value)])
                else:
                    result.append(str(value))
            elif type(value) is str:
                result.append(' '*spaces*(depth)+' '*(key_len+1))
                result.append(value,styles['str'])
            else:
                result.append("<" + str(type(value))[8:-2] + ", ID:" + str(id(value)) + ">...(Cannot Unpack)",styles.setdefault(type(Object),'obj'))
        result.append('\n')
        return result

    @staticmethod
    def objtext(obj, depth=float('inf'), spaces=4, save=False,styles:dict = __styles) -> Text:
        result = SuperObj.Text()
        result = SuperObj.__unpackClass( 
                    objID=id(obj),
                    result=result,
                    spaces=spaces,
                    max_depth=depth,
                    loop_check = [],
                    styles = styles
                )
        if save:
            result = str(result)
            for i in SuperObj.__styles.values():
                result = result.replace(i, '')
            with open(f'./{id(obj)}.txt', 'w') as f:
                f.write(result)
        return result

    @staticmethod
    def objprint(obj, depth=float('inf'), spaces=4, notes=True,styles:dict = dict(),**changed_styles) -> None:
        from rich.console import Console
        console = Console(markup=False,highlight=False)
        info = SuperObj.Text()
        styles = SuperDict.merge(SuperObj.__styles,changed_styles)
        console.print(SuperObj.objtext(obj, depth, spaces,styles=styles))
        if notes:
            for num, i in enumerate(styles):
                if isinstance(i, str):
                    k = str(i)
                elif i is None:
                    k = 'None'
                else:
                    k = str(i)[8:-2]
                if num % 2 == 0:
                    info.append(f'■ :{k}'+' '*(10-len(k)),style=styles[i])
                else:
                    info.append(f'■ :{k}\n',style=styles[i])
        console.print(info)
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