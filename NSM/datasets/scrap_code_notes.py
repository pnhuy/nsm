def test(**kwargs):
    for kwarg in kwargs:
        print(kwarg)
        print(kwargs[kwarg])

dict_ = {'one': 1, 'two': 2}
test(**dict_)


x = ['one_1', 'one_4', 'one_3, one_0', 'one_2']
sorted_x = sorted(x, key=lambda s: int(s.split('_')[-1]))
