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

## Sections

- [Mathematical Properties](#mathematical-properties)
- [Parser](#parser)
  - [Parser combinators](#parser-combinators)
  - [Lexer](#lexer)
  - [Tree](#tree)
  - [Parser Implementation](#parser-implementation)
- [Reducing terms](#reducing-terms)
- [Symmetry](#symmetry)
- [Distributivity](#distributivity)
- [Associativity](#associativity)
- [Rewriting](#rewriting)
- [Unifiying/Equating](#unifiyingequating)
- [Final](#final)

## Mathematical Properties

To write a equation solver, we need to clarify what are the mathematical properties
we are working with. The main ones that we are going to use are the
`Associativity`, `Identity`, `Commutativity`, `Symmetry`, `Distribitivity`. We can
define the following rules:

- `Associativity`: (a + b) + c = a + (b + c)
- `Identity`: a = a
- `Commutativity`: a + b = b + a
- `Symmetry`: a + b = c; a = c - b

Ok, right, these mathematical properties are hard to read if we don't have any
code to parallel with it, so let's start writing the parser.

## Parser

We need first to tokenize the inputs into a bunch of tokens, which are a kind
of letter with spaces, and some characters ignored:

```rs
/// The token type used by the lexer.
#[derive(Debug, PartialEq, Clone)]
pub enum Token<'src> {
    Number(usize),
    Decimal(usize, usize),
    Identifier(&'src str),
    Ctrl(char),
}
```

The token tooks the lifetime `src`, because it's referring directly the part
of the source code.

### Parser Combinators

We are using technique called [parser combinator](https://en.wikipedia.org/wiki/Parser_combinator). And we are using a library [chumsky](https://github.com/zesterer/chumsky) to write parser combinators.

### Lexer

The code of the lexer is simply, removing the junk and returning the letters and
numbers as `Token`. Read the snippet

```rs
/// Parse a string into a set of tokens.
///
/// This function is a wrapper around the lexer and parser, and is the main entry point
/// for the calculator.
fn lexer<'src>() -> impl Parser<'src, &'src str, Vec<(Token<'src>, Span)>, LexerError<'src>> {
    let num = text::int(10)
        .then(just('.').then(text::int(10)).or_not())
        .try_map(|(int, decimal), span| {
            // int: &str, decimal: Option<(char, &str)>
            // define the types of the variables, because the chumsky
            // parser tries to infer it
            let int: &str = int;
            let decimal: Option<(char, &str)> = decimal;

            let Ok(int) = int.parse::<usize>() else {
                return Err(Rich::custom(span, "invalid integer"));
            };
            let Some((_, decimal)) = decimal else {
                return Ok(Token::Number(int));
            };

            let Ok(decimal) = decimal.parse::<usize>() else {
                return Err(Rich::custom(span, "invalid decimal"));
            };

            Ok(Token::Decimal(int, decimal))
        })
        .labelled("number");

    // Maps the common mathematical operations like addition, multiplication
    // subtraction, division, and go on..
    let op = one_of("+*-/!^|&<>=")
        .repeated()
        .at_least(1)
        .map_slice(Token::Identifier)
        .labelled("operator");

    // Maps simple incognito variables into identifiers, these are the variables
    // we are trying to discover :)
    let ident = text::ident().map(Token::Identifier).labelled("icognito");

    // The groups that change the precedence.
    let ctrl = one_of("()[]{}").map(Token::Ctrl).labelled("ctrl");

    // Now this finishes the lexer
    num.or(op)
        .or(ctrl)
        .or(ident)
        .map_with_span(|token, span| (token, span))
        .padded()
        .repeated()
        .collect()
}
```

> The full source code is wrote in [main.rs](https://github.com/aripiprazole/eq/blob/main/src/main.rs#L274).

### Tree

We need to define an abstract syntax tree, to translate the mathematical terms
into it:

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

Define the variables and incognitos:

```rs
/// A variable. This is a variable that can be assigned to.
#[derive(Default, Debug, Clone)]
pub struct Variable {
    pub name: String,

    /// We use Rc of RefCell here so we can clone and use internal
    /// mutability to change it's value
    pub data: Rc<RefCell<Option<Term>>>,
}

impl Variable {
    /// Gets the data of the variable.
    pub fn data(&self) -> Option<Term> {
        *self.data.borrow().deref()
    }
}

impl PartialEq for Variable {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Variable {}

impl Hash for Variable {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}
```

And finally, the binary operations:

```rs
/// A binary operation. This is a binary operation that takes two arguments.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct BinOp {
    pub op: Op,
    pub lhs: Term,
    pub rhs: Term,
}
```

The full source code can be found in the [main.rs](https://github.com/aripiprazole/eq/blob/main/src/main.rs#L117).

### Parser implementation

Now we need to write an expression parser, which will take translate tokens into
mathematical terms.

```rs
recursive(|expr| {
    // Defines the parser for the value. It is the base of the
    // expression parser.
    let value = select! {
            Token::Number(number) => Expr::Number(number),
            Token::Identifier(identifier) => Expr::Variable(Variable { name: identifier.into(), data: Rc::default() }),
        }
        .map_with_span(|kind, span| (kind, span))
        .map_with_state(|term, _, state: &mut TermArena| state.intern(term))
        .labelled("value");
});
```

Note thate we used state in the parser, and this is the "arena". The arena will
store the real expressions, and will return an ID to the expression. We will have
to define the `Term`:

```rs
#[derive(Default, Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Term(usize);
```

And define the "arena":

```rs
#[derive(Default)]
pub struct TermArena {
    pub id_to_slot: HashMap<Term, Rc<Spanned<Expr>>>,
    pub slot_to_id: HashMap<Expr, Term>,
}

impl TermArena {
    /// Creates a new term in the arena
    pub fn intern(&mut self, term: Spanned<Expr>) -> Term {
        let id = Term(fxhash::hash(&term.0));
        self.id_to_slot.insert(id, Rc::new(term.clone()));
        self.slot_to_id.insert(term.0, id);
        id
    }

    /// Checks if the term exists in the arena.
    pub fn exists(&self, term: &Expr) -> Option<Term> {
        self.slot_to_id.get(term).cloned()
    }

    /// Gets the term from the arena.
    pub fn get(&self, id: Term) -> Rc<Spanned<Expr>> {
        self.id_to_slot
            .get(&id)
            .cloned()

            // It will return a default expression so the program doesn't crash
            // when the id is missing, it's called sentinel values
            .unwrap_or_else(|| Rc::new((Expr::Error, Range::<usize>::default().into())))
    }
}
```

> Here it's used the library [fxhash](https://crates.io/crates/fxhash) to fast
> hash the values

The arena stuff makes interning of expressions, which is returning the same ID
to same-hash expressions, if the expressions have the same hash, they will return
the same hash, so they will return the same ID. This is useful to improve
performance, and later doing the `Commutativvity` rule, since `1 + 2` is the
same as `2 + 1`, and the hash of it is the same.

So now, we need to write the group terms, like [], {} or even (), so we are
writing the following code in the `recursive` function.

> This is why we need to use recursive, the code will refer to expression

```rs
let brackets = expr
    .clone()
    .delimited_by(just(Token::Ctrl('[')), just(Token::Ctrl(']')));

let parenthesis = expr
    .clone()
    .delimited_by(just(Token::Ctrl('(')), just(Token::Ctrl(')')));

let braces = expr
    .clone()
    .delimited_by(just(Token::Ctrl('{')), just(Token::Ctrl('}')));
```

And we define the "primary" expression, which will catch all values, and
group expressions.

```rs
// Defines the parser for the primary expression. It is the
// base of the expression parser.
let primary = value
    .or(braces)
    .or(parenthesis)
    .or(brackets)
    .labelled("primary");
```

Now we need to define mathematical operations like addition, subtraction,
multiplication and division.

They are splitted in two declarations, so we can have precedence.

```rs
let factor = primary
    .foldl_with_state(
        just(Token::Identifier("+"))
            .to(Op::Add)
            .or(just(Token::Identifier("-")).to(Op::Sub))
            .then(expr.clone())
            .repeated(),
        |lhs: Term, (op, rhs), state: &mut TermArena| {
            let (_, fst) = &*state.get(lhs);
            let (_, snd) = &*state.get(rhs);

            let span = SimpleSpan::new(fst.start, snd.end);
            let expr = Expr::BinOp(BinOp { op, lhs, rhs });

            // RULE: Commutativity
            //
            // The commutativity rule states that the order of the operands
            // does not matter. This means that `1 + 2` is the same as `2 + 1`.
            //
            // This rule is implemented by checking if the expression already
            // exists in the state. If it does, then we return the existing
            // expression, otherwise we create a new one.
            match state.exists(&Expr::BinOp(BinOp { op, lhs: rhs, rhs: lhs })) {
                Some(term) if op == Op::Add => term,
                None => state.intern((expr, span)),
                _ => state.intern((expr, span)),
            }
        },
    )
    .labelled("factor");

let term = factor
    .foldl_with_state(
        just(Token::Identifier("*"))
            .to(Op::Mul)
            .or(just(Token::Identifier("/")).to(Op::Div))
            .then(expr.clone())
            .repeated(),
        |lhs: Term, (op, rhs), state: &mut TermArena| {
            let (_, fst) = &*state.get(lhs);
            let (_, snd) = &*state.get(rhs);

            let span = SimpleSpan::new(fst.start, snd.end);
            let expr = Expr::BinOp(BinOp { op, lhs, rhs });

            // RULE: Commutativity
            //
            // The commutativity rule states that the order of the operands
            // does not matter. This means that `1 + 2` is the same as `2 + 1`.
            //
            // This rule is implemented by checking if the expression already
            // exists in the state. If it does, then we return the existing
            // expression, otherwise we create a new one.
            match state.exists(&Expr::BinOp(BinOp { op, lhs: rhs, rhs: lhs })) {
                Some(term) if op == Op::Mul => term,
                None => state.intern((expr, span)),
                _ => state.intern((expr, span)),
            }
        },
    )
    .labelled("term");
```

> The commutativity part starts here, the parser will check if the expression
> already exists in the context, and will return the same id, if it exists
>
> Of course, this isn't an automatic process, so we need to reverse the operators
> and try to find it as reversed, and if it does exist, we take it from the
> context.

Now, we have finished the `expression` parser, and you can have a look in the
full source code [here](https://github.com/aripiprazole/eq/blob/main/src/main.rs#L336).
We must write the equation parser, and it's quite simple to do it.

```rs
// Defines the parser for the equation. It is the base of the
// parser of equations and inequations.
expr_parser
    .clone()
    .then(
        // Parses an operation
        just(Token::Identifier("="))
            .to(EqKind::Eq)
            .or(just(Token::Identifier("!=")).to(EqKind::Neq)),
    )
    .then(expr_parser.clone())
    .map(|((lhs, op), rhs)| Equation { kind: op, lhs, rhs })
    .map_with_span(|equation, span| (equation, span))
    .labelled("equation")
```

Its the combination of `expr (== | !=) expr`.

## Reducing terms

We need to reduce the terms to it's normal form, to be compared with another
terms, like: `1 + 2` needs to be reduced to `3` to be compared with `3` properly.

It's the first rewrite rule we need to write! So let's create a function like
`rewrite` in

```rs
impl Term {
    /// Rewrites a term to its normal form.
    pub fn rewrite(self, state: &mut TermArena) -> Term {
        self.normalize(state)
    }
}
```

And start writing the [`normalize function`](https://wiki.haskell.org/Weak_head_normal_form).
which is the reduced/or evaluated form.

The link is appointing to the Haskell documentation.

```rs
impl Term {
    /// Reduces a term to its normal form.
    pub fn normalize(self, state: &mut TermArena) -> Term {
        let (kind, span) = &*state.get(self);
        let new_kind = match kind {
            Expr::Group(group) => Expr::Group(group.rewrite(state)),
            Expr::BinOp(bin_op) => {
                let lhs = bin_op.lhs.rewrite(state);
                let rhs = bin_op.rhs.rewrite(state);

                // If the term is a number, we try to reduce it evaluating
                // the operation.
                match &state.get(lhs).0 {
                    // If the term is a number
                    //
                    // We assume that the term is a number and we try to
                    // reduce it.
                    Expr::Number(lhs) => match &state.get(rhs).0 {
                        Expr::Number(rhs) => {
                            let number = match bin_op.op {
                                Op::Add => lhs + rhs,
                                Op::Sub => lhs - rhs,
                                Op::Mul => lhs * rhs,
                                Op::Div => lhs / rhs,
                            };

                            Expr::Number(number)
                        }
                        Expr::Group(group) => return group.rewrite(state),
                        _ => return self,
                    },
                    Expr::Group(group) => return group.rewrite(state),
                    _ => return self,
                }
            }
            // If the term is a variable, we try to reduce it.
            Expr::Variable(hole) => match hole.data() {
                Some(value) => return value.rewrite(state),
                None => kind.clone(),
            },
            _ => kind.clone(),
        };
        state.intern((new_kind, *span))
    }
}
```

It basically evaluates the term to it's normal form

## Symmetry

We need to write the symmetry rule, which will be used in a further step called
`unifying/equating`.

```rs
impl BinOp {
    /// RULE: Symmetry
    ///
    /// Examples:
    /// `a + b = c`
    /// `a = c - b`
    pub fn symmetry(&self, value: Term, state: &mut TermArena) -> (Term, Term) {
        let reverse_op = match self.op {
            Op::Add => Op::Sub,
            Op::Sub => Op::Add,
            Op::Mul => Op::Div,
            Op::Div => Op::Mul,
        };

        let span: Span = (state.get(self.lhs).1.start..state.get(self.rhs).1.end).into();

        let lhs = state.intern((
            Expr::BinOp(BinOp {
                op: reverse_op,
                lhs: value,
                rhs: self.rhs,
            }),
            span,
        ));

        (lhs.rewrite(state), self.lhs.rewrite(state))
    }
}
```

The symmetry is basically, the following steps with the given [BinOp]:

```
x + 7 = 10
x = 10 - 7
```

It's a fundamental step for solving equations!

## Distributivity

The distributivity property is when we apply an operation to it's left side, like

```
(x + 2) * 2
```

Which will be rewrote into:

```
2 * x + 4
```

This is fundamental to write some equations, the source code for this step is:

```rs
impl Term {
    /// Distributes a term over another term.
    pub fn distribute(self, op: Op, another: Term, state: &mut TermArena) -> Term {
        let (kind, span) = &*state.get(self);
        let new_kind = match kind {
            Expr::Group(_) => return self,
            Expr::BinOp(bin_op) => {
                let lhs = bin_op.lhs.apply_distributive_property(state);
                let rhs = bin_op.rhs.apply_distributive_property(state);

                Expr::BinOp(BinOp {
                    op: bin_op.op,
                    lhs,
                    rhs,
                })
            }
            _ => Expr::BinOp(BinOp {
                op,
                lhs: self,
                rhs: another,
            }),
        };

        state.intern((new_kind, *span))
    }

    /// Applies the distributive property to a term.
    pub fn apply_distributive_property(self, state: &mut TermArena) -> Term {
        let Expr::BinOp(bin_op) = &state.get(self).0 else {
            return self;
        };

        match (&state.get(bin_op.lhs).0, &state.get(bin_op.rhs).0) {
            (Expr::Group(group), _) => group.distribute(bin_op.op, bin_op.rhs, state),
            (_, Expr::Group(group)) => group.distribute(bin_op.op, bin_op.lhs, state),
            (_, _) => self,
        }
    }
}
```

## Associativity

The associativity rules represents an equation that's like: (1 + 2) + 3 = 1 + (2 + 3). It does represents an reorder based in the precedence, to normalize the operation.

We can write a code to associate 3 terms based on it's operator precedence:

```rs
/// Applies the associativity rule to a binary operation.
fn associate(lhs: Term, fop: Op, mhs: Term, sop: Op, rhs: Term, state: &mut TermArena) -> BinOp {
    // RULE: Associativity
    //
    // If the term is a binary operation, we try to reduce it
    // to its weak head normal form using the precedence of
    // the operators.
    //
    // This step is called precedence climbing.
    //
    // Evaluate the operation if the precedence of the
    // operator is higher than the precedence of the
    // operator of the right hand side.
    let lhs = lhs.apply_associativity(state).rewrite(state);
    let mhs = mhs.apply_associativity(state).rewrite(state);
    let rhs = rhs.apply_associativity(state).rewrite(state);

    // If the precedence of the operator of the left hand side
    // is higher than the precedence of the operator of the
    // right hand side, we change the order.
    if op_power(sop) >= op_power(fop) {
        BinOp {
            op: fop,
            lhs,
            rhs: state.intern((
                Expr::BinOp(BinOp {
                    op: sop,
                    lhs: mhs,
                    rhs,
                }),
                (0..0).into(),
            )),
        }
    } else {
        BinOp {
            op: fop,
            lhs: state.intern((
                Expr::BinOp(BinOp {
                    op: sop,
                    lhs,
                    rhs: mhs,
                }),
                (0..0).into(),
            )),
            rhs,
        }
    }
}
```

And write a wrapper in [Term] to call it with [BinOp] operations:

```rs
impl Term {
    /// Applies the associativity rule to a term.
    pub fn apply_associativity(self, state: &mut TermArena) -> Term {
        let (kind, span) = &*state.get(self);

        // If the term is not a binary operation, we return it.
        let Expr::BinOp(mut bin_op) = kind.clone() else {
            return self;
        };

        // Apply associativy to the leftmost side of the expression.
        //
        // This is done by recursively applying the associativity
        if let Expr::BinOp(lhs_bin) = &state.get(bin_op.lhs).0 {
            bin_op = associate(
                lhs_bin.lhs,
                lhs_bin.op,
                lhs_bin.rhs,
                bin_op.op,
                bin_op.rhs,
                state,
            );
        }

        // Apply associativy to the rightmost side of the expression.
        //
        // This is done by recursively applying the associativity
        if let Expr::BinOp(rhs_bin) = &state.get(bin_op.rhs).0 {
            bin_op = associate(
                bin_op.lhs,
                bin_op.op,
                rhs_bin.lhs,
                rhs_bin.op,
                rhs_bin.rhs,
                state,
            );
        }

        // Reintern the term.
        state.intern((Expr::BinOp(bin_op), *span))
    }
}
```

## Rewriting

We need to change the `rewrite` function to compute all rewrite rules we have
made:

```rs
/// Rewrites a term to its normal form.
pub fn rewrite(self, state: &mut TermArena) -> Term {
    self.apply_distributive_property(state)
        .apply_associativity(state)
        .normalize(state)
}
```

## Unifiying/Equating

Now we need to write the logical part, which will compare the operations. We need
to first start a pattern matching:

```rs
impl Term {
    /// Unifies two terms. It's the main point of the equation.
    ///
    /// The logic relies here.
    pub fn unify(self, another: Term, state: &mut TermArena) -> Result<(), TypeError> {
        match (&state.get(self).0, &state.get(another).0) {
            // Errors are sentinel values, so they are threated like holes
            // and they are ignored, anything unifies with them.
            (Expr::Error, _) => {}
            (_, Expr::Error) => {}

            (a, b) => {
                return Err(TypeError::NotUnifiable(a.clone(), b.clone()));
            }
            // ...
        }

        Ok(())
    }
}
```

> The terms successfully unified will fallback ino the `Ok(())` expressions,
> and if it's not unified, it will return an error just like in the `TypeError::NotUnifiable`
> part

And wrap the group terms, unifying its values:

```rs
// Reduce groups to the normal form
(_, Expr::Group(another)) => {
    let term = self.rewrite(state);
    let another = another.rewrite(state);

    term.unify(another, state)?;
}
(Expr::Group(term), _) => {
    let term = term.rewrite(state);
    let another = another.rewrite(state);

    term.unify(another, state)?;
}
```

And unify numbers, if they are the same number, it will unify properly

```rs
// If they are the same, they unify.
(Expr::Number(a), Expr::Number(b)) if a == b => {}
```

Now, the variable stuff, which is the same as the number part, if the
name is the same, it will unify.

```rs

// If they are variables, they unify if they are the same.
//
// This check isn't inehenterly necessary, but it's a good catcher
// to avoid panics, because if the variables are the same, they will
// try to borrow the same data, and it will panic with ref cells.
(Expr::Variable(variable_a), Expr::Variable(variable_b))
    if variable_a.name == variable_b.name => {}
```

Now we need to unify attributions, like `x = 5`, or something like this, this
will basically give the variable, a meaning.

```rs
// Unifies the variable with the term, if the variable is not bound. If
// it's bound, it will try to unify, if it's not unifiable, it will
// return an error.
(_, Expr::Variable(variable)) => {
    match variable.data() {
        // If the variable is already bound, we unify the bound
        Some(bound) => {
            self.unify(bound, state)?;
        }
        // Empty hole
        None => {
            variable.data.replace(Some(self));
        }
    }
}
(Expr::Variable(variable), _) => {
    match variable.data() {
        // If the variable is already bound, we unify the bound
        Some(bound) => {
            bound.unify(another, state)?;
        }
        // Empty hole
        None => {
            variable.data.replace(Some(another));
        }
    }
}

```

And now, unify the operations, like `1 + 1` = `1 + 1`, we need to reduce the
terms, like: `1 + 1` is equivalent to `2`, so we will compare `2` with `2`,
and it will successfully unify:

```rs
// Unifies the bin ops if they are the same, and unifies the
// operands.
(Expr::BinOp(bin_op_a), Expr::BinOp(bin_op_b)) => {
    if bin_op_a.op == bin_op_b.op {
        let lhs_a = bin_op_a.lhs.rewrite(state);
        let rhs_a = bin_op_a.rhs.rewrite(state);

        let lhs_b = bin_op_b.lhs.rewrite(state);
        let rhs_b = bin_op_b.rhs.rewrite(state);

        lhs_a.unify(lhs_b, state)?;
        rhs_a.unify(rhs_b, state)?;
    } else {
        return Err(TypeError::IncompatibleOp(bin_op_a.op, bin_op_b.op));
    }
}
```

Now, the last part, we need to unify the symmetry, if we have an example like
the comments, it will use the `symmetry` function.

```rs
// If the term is a number, we try the following steps given the example:
//   9 = x + 6
//
// 1. We get the reverse of `+`, which is `-`.
// 2. We subtract `6` from both sides, to equate it,
//    and since the `9` is a constant, we can reduce
//    it to the normal form.
// 3. Got the equation solved, we can unify the `x` with
//    the result of the subtraction.
//
//    3 = x and then x = 3
(Expr::Number(_), Expr::BinOp(bin_op)) => {
    let (term, another) = bin_op.symmetry(self, state);

    term.unify(another, state)?;
}
(Expr::BinOp(bin_op), Expr::Number(_)) => {
    let (term, another) = bin_op.symmetry(another, state);

    term.unify(another, state)?;
}
```

# Final

Thanks for your read :) Have a nice day!
