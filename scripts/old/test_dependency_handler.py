
def modelA(propertyB=B,
           propertyC=C):
    return B + C


def modelB(propertyD=D):
    return D


D = 5
C = 2
B = modelB(D)
A = modelA(B, C)

properties = {}
properties["D"] = D
properties["C"] = C
properties["B"] = B
properties["A"] = A

models = {
    "A": {
        "model": modelA,
        "propertyB": "B",
        "propertyC": "C",
    },
    "B": {
        "model": modelB,
        "propertyD": "D"}
    },
}