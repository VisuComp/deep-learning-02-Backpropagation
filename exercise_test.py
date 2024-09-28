
import pytest
from pytest import approx
import torch

from IPython.display import display, HTML

# Define the HTML and CSS for the green info box with a smiley
message = """
<div style="
    padding: 10px;
    border-radius: 5px;
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    font-size: 16px;
    font-family: Arial, sans-serif;
    margin: 10px 0;
">
    <span style="font-size: 20px;">&#128512;</span>  <!-- Smiley emoji -->
    <strong>Gut gemacht!</strong> <!-- Text -->
</div>
"""

# Aufgabe 1

def test_aufgabe1(Autograd_und_backprop):
    # Check optimizer is of type SGD
    assert isinstance(Autograd_und_backprop['optim'], torch.optim.SGD), "Aufgabe 1: optim ist nicht vom Typ torch.optim.SGD."

    # check Learning rate 
    assert Autograd_und_backprop['optim'].param_groups[0]['lr'] == approx(1e-3), "Aufgabe 1: Learning rate bei optim nicht richtig."

    # Check optimizer optimizes two parameters
    assert len(Autograd_und_backprop['optim'].param_groups[0]['params'])==2, "Aufgabe 1: param optimiert nicht beide Parameter w,b"
    display(HTML(message))

# Aufgabe 2

def test_aufgabe2(Autograd_und_backprop):
    assert 'erwartet_w' in Autograd_und_backprop, "Aufgabe 2: Variable erwartet_w nicht deklariert"
    assert 'erwartet_b' in Autograd_und_backprop, "Aufgabe 2: Variable erwartet_b nicht deklariert"

    assert(Autograd_und_backprop['erwartet_w'] == approx(2 - (1E-3 * 3 * 2))), "Aufgabe 2: Falscher Wert für erwartet_w"
    assert(Autograd_und_backprop['erwartet_b'] == approx(1 - (1E-3 * 1 * 2))), "Aufgabe 2: Falscher Wert für erwartet_b"
    display(HTML(message))

# Aufgabe 3
def test_aufgabe3(Autograd_und_backprop):
    assert 'w' in Autograd_und_backprop, "Aufgabe 3: Variable w nicht deklariert"
    assert 'b' in Autograd_und_backprop, "Aufgabe 3: Variable b nicht deklariert"

    assert isinstance(Autograd_und_backprop['w'], torch.Tensor), "Falscher Datentyp für Variable w (erwartet: torch.Tensor)"
    assert isinstance(Autograd_und_backprop['b'], torch.Tensor), "Falscher Datentyp für Variable b (erwartet: torch.Tensor)"
    assert Autograd_und_backprop['w'].detach() == approx(2 - (1E-3 * 3 * 2)), "Aufgabe 3: Falscher Wert für w"
    assert Autograd_und_backprop['b'].detach() == approx(1 - (1E-3 * 1 * 2)), "Aufgabe 3: Falscher Wert für b"
    display(HTML(message))
