y = 0

def test_read(x):
    # x = x
    print(x)
    if 'x' in globals():
        print('x in globals')
    if 'x' in locals():
        print('x in locals')

    print(f'y={y}')

# test_read(1)