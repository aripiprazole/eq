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

### Real Parser

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