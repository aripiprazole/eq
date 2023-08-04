# Writing an Equation Solver

Writing an Equation Solver is a process that is made of: parsing,
equating/unifying and rewriting.

- Equating: it's the step that two terms are equalized and
tries to make equity between them 2, just like an equation

- Rewriting: to write equations in real life we need to rewrite
the equation until the variables are discovered, right? For an example:
```
10 = x + 7
10 - 7 = x + 7 - 7
3 = x
x = 3
```
So, there was 3 rewrites before the final result. The same process is
applied in a computer, we need to apply some mathematical properties and
rewrite it, and normalise it.

- Parsing: we need to get the string of the equation and translate into
objects like the following enum:
```rs
/// A term in the calculator. This is the lowest level of the calculator.
#[derive(Default, Debug, Clone, Hash, PartialEq, Eq)]
pub enum Expr {
    #[default]
    Error,
    Number(usize),
    Group(Term),
    Variable(Variable),
    BinOp(BinOp),
}
```

## Mathematical Properties

To write a equation solver, we need to clarify what are the mathematical properties
we are working with. The main ones that we are going to use are the
`Associativity`, `Identity`, `Commutativity`, `Symmetry`, `Distribitivity`. We can
define the following rules:

- `Associativity`: (a + b) + c = a + (b + c)
- `Identity`: a = a
- `Commutativity`: a + b = b + a
- `Symmetry`: a + b = c; a = c - b
