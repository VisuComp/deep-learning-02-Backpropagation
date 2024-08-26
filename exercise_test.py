
import pytest
from pytest import approx
import torch


from importnb import imports
with imports("ipynb"):
    import Autograd_und_backpropagation as Autograd_und_backprop


# Aufgabe 1

def test_aufgabe1():
    # Check optimizer is of type SGD
    assert isinstance(Autograd_und_backprop.optim, torch.optim.SGD), "Aufgabe 1: optim ist nicht vom Typ torch.optim.SGD."

    # check Learning rate 
    assert Autograd_und_backprop.optim.param_groups[0]['lr'] == approx(1e-3), "Aufgabe 1: Learning rate bei optim nicht richtig."

    # Check optimizer optimizes two parameters
    assert len(Autograd_und_backprop.optim.param_groups[0]['params'])==2, "Aufgabe 1: param optimiert nicht beide Parameter w,b"

# Aufgabe 2

def test_aufgabe2():
    assert hasattr(Autograd_und_backprop, 'erwartet_w'), "Aufgabe 2: Variable erwartet_w nicht deklariert"
    assert hasattr(Autograd_und_backprop, 'erwartet_b'), "Aufgabe 2: Variable erwartet_b nicht deklariert"

    assert(Autograd_und_backprop.erwartet_w == approx(2 - (1E-3 * 3 * 2))), "Aufgabe 2: Falscher Wert für erwartet_w"
    assert(Autograd_und_backprop.erwartet_b == approx(1 - (1E-3 * 1 * 2))), "Aufgabe 2: Falscher Wert für erwartet_b"

# Aufgabe 3
def test_aufgabe3():
    assert hasattr(Autograd_und_backprop, 'w'), "Aufgabe 3: Variable w nicht deklariert"
    assert hasattr(Autograd_und_backprop, 'b'), "Aufgabe 3: Variable b nicht deklariert"

    assert isinstance(Autograd_und_backprop.w, torch.Tensor), "Falscher Datentyp für Variable w (erwartet: torch.Tensor)"
    assert isinstance(Autograd_und_backprop.b, torch.Tensor), "Falscher Datentyp für Variable b (erwartet: torch.Tensor)"
    assert Autograd_und_backprop.w.detach() == approx(2 - (1E-3 * 3 * 2)), "Aufgabe 3: Falscher Wert für w"
    assert Autograd_und_backprop.b.detach() == approx(1 - (1E-3 * 1 * 2)), "Aufgabe 3: Falscher Wert für b"
