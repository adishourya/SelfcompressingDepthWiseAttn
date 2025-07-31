# next neareset 32
def next_nearest32(n:int)-> int:
    return ( (n+31)// 32 ) * 32
