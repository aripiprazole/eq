use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::{Debug, Display},
    hash::Hash,
    ops::{Deref, Range},
    rc::Rc,
};

use ariadne::{Color, Label, Report, ReportKind, Source};
use chumsky::{
    extra::{self, Err},
    prelude::{Input, Rich},
    primitive::{just, one_of},
    recursive::recursive,
    select,
    span::SimpleSpan,
    text, IterParser, Parser,
};
use fxhash::FxBuildHasher;
use im_rc::HashSet;

/// The token type used by the lexer.
#[derive(Debug, PartialEq, Clone)]
pub enum Token<'src> {
    Number(usize),
    Decimal(usize, usize),
    Identifier(&'src str),
    Ctrl(char),
}

impl<'src> Display for Token<'src> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Number(value) => write!(f, "{value}"),
            Token::Decimal(int, decimal) => write!(f, "{int}.{decimal}"),
            Token::Identifier(id) => write!(f, "{id}"),
            Token::Ctrl(ctrl) => write!(f, "{ctrl}"),
        }
    }
}

/// An operator. This is a binary operation that takes two arguments.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

/// A binary operation. This is a binary operation that takes two arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinOp {
    pub op: Op,
    pub lhs: Term,
    pub rhs: Term,
}

impl BinOp {
    /// RULE: Symmetry
    ///
    /// Examples:
    /// `a + b = b + a`
    /// `a - b = b - a`
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

impl Hash for BinOp {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self.op {
            // Commutative
            //
            // If the operator is commutative, we don't need to take the order
            //
            // TODO: Fixme i think it does get the order into account
            Op::Add | Op::Mul => {
                self.op.hash(state);
                let mut hash_set: HashSet<Term, FxBuildHasher> = HashSet::default();
                hash_set.insert(self.lhs);
                hash_set.insert(self.rhs);
                hash_set.hash(state);
            }
            // Non-commutative
            //
            // If the operator is non-commutative, we need to make sure that
            // the order of the operands is taken into account when hashing.
            Op::Div | Op::Sub => {
                self.op.hash(state);
                state.write_i8(1);
                self.lhs.hash(state);
                state.write_i8(2);
                self.rhs.hash(state);
            }
        }
    }
}

/// A variable. This is a variable that can be assigned to.
#[derive(Default, Debug, Clone)]
pub struct Variable {
    pub name: String,
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

impl Term {
    /// Creates a debug wrapper for the term, which will print
    /// the term in a human-readable format.
    pub fn debug(self, arena: &TermArena) -> ExprDebug {
        self.debug_with_fuel(128, arena)
    }

    /// Creates a debug wrapper for the term, which will print
    /// the term in a human-readable format.
    ///
    /// With specified fuel.
    pub fn debug_with_fuel(self, fuel: usize, arena: &TermArena) -> ExprDebug {
        ExprDebug {
            arena,
            term: self,
            fuel,
        }
    }
}

/// A debug wrapper for a term, which will print the term in a human-readable format.
#[derive(Copy, Clone)]
pub struct ExprDebug<'a> {
    arena: &'a TermArena,
    term: Term,

    /// The amount of fuel to use when printing the term.
    ///
    /// The fuel is used to prevent infinite recursion when printing
    /// the term. If the fuel runs out, the term will be printed as `…`.
    fuel: usize,
}

impl Debug for ExprDebug<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", show(self.term, self.fuel, self.arena))
    }
}

/// Shows a term in a human-readable format.
pub fn show(term: Term, fuel: usize, state: &TermArena) -> String {
    if fuel == 0 {
        return "…".into();
    }

    match &state.get(term).0 {
        Expr::Error => "error".into(),
        Expr::Number(n) => format!("{n}"),
        Expr::Group(group) => format!("({})", show(*group, fuel - 1, state)),
        Expr::Variable(variable) => match variable.data() {
            Some(value) => format!("{}", show(value, fuel - 1, state)),
            None => format!("?{}", variable.name),
        },
        Expr::BinOp(bin_op) => {
            let lhs = show(bin_op.lhs, fuel - 1, state);
            let rhs = show(bin_op.rhs, fuel - 1, state);
            let op_str = match bin_op.op {
                Op::Add => "+",
                Op::Sub => "-",
                Op::Mul => "*",
                Op::Div => "/",
            };

            format!("({lhs} {op_str} {rhs})")
        }
    }
}

#[derive(Default, Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Term(usize);

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

    /// Gets the term from the arena.
    pub fn get(&self, id: Term) -> Rc<Spanned<Expr>> {
        self.id_to_slot
            .get(&id)
            .cloned()
            .unwrap_or_else(|| Rc::new((Expr::Error, Range::<usize>::default().into())))
    }

    /// Gets the term from the arena.
    pub fn resolutions(&self) -> Vec<(String, Term)> {
        self.id_to_slot
            .iter()
            .filter_map(|(_, slot)| {
                if let Expr::Variable(variable) = &slot.0 {
                    variable.data().map(|value| (variable.name.clone(), value))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    }
}

/// The kind of equation. This is used to determine the kind of comparison to perform.
///
/// It can be an equality or inequality.
#[derive(Debug, Clone, Copy)]
pub enum EqKind {
    Neq,
    Eq,
}

/// An equation. This is the top-level type of the calculator.
#[derive(Debug, Clone)]
pub struct Equation {
    pub kind: EqKind,
    pub lhs: Term,
    pub rhs: Term,
}

//// The span type used by the parser.
type Span = SimpleSpan;

/// The input type for the parser.
type ParserInput<'tokens, 'src> =
    chumsky::input::SpannedInput<Token<'src>, Span, &'tokens [(Token<'src>, Span)]>;

/// A lexer error.
type LexerError<'src> = Err<Rich<'src, char, Span>>;

/// A spanned value. A value that has a source code span attached to it.
type Spanned<T> = (T, Span);

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

    let op = one_of("+*-/!^|&<>=")
        .repeated()
        .at_least(1)
        .map_slice(Token::Identifier)
        .labelled("operator");

    let ident = text::ident().map(Token::Identifier).labelled("icognito");

    let ctrl = one_of("()[]{}").map(Token::Ctrl).labelled("ctrl");

    num.or(op)
        .or(ctrl)
        .or(ident)
        .map_with_span(|token, span| (token, span))
        .padded()
        .repeated()
        .collect()
}

/// Defines the base parser for the simple math language. It
/// does parse a set of tokens, into a equation.
///
/// The parser is defined as a function, because it is recursive.
///
/// [`recursive`]: https://docs.rs/chumsky/0.1.0/chumsky/recursive/index.html
/// [`Parser`]: https://docs.rs/chumsky/0.1.0/chumsky/prelude/trait.Parser.html
/// [`Expr`]: [`Expr`]
/// [`Equation`]: [`Equation`]
fn parser<'tokens, 'src: 'tokens>() -> impl Parser<
    // Input Types
    'tokens,
    ParserInput<'tokens, 'src>,
    Spanned<Equation>,
    extra::Full<Rich<'tokens, Token<'src>, Span>, TermArena, ()>,
> {
    // Defines the parser for the expression. It is recursive, because
    // it can be nested.
    let expr_parser = recursive(|expr| {
        // Defines the parser for the value. It is the base of the
        // expression parser.
        let value = select! {
                Token::Number(number) => Expr::Number(number),
                Token::Identifier(identifier) => Expr::Variable(Variable { name: identifier.into(), data: Rc::default() }),
            }
            .map_with_span(|kind, span| (kind, span))
            .map_with_state(|term, _, state: &mut TermArena| state.intern(term))
            .labelled("value");

        let brackets = expr
            .clone()
            .delimited_by(just(Token::Ctrl('[')), just(Token::Ctrl(']')));

        let parenthesis = expr
            .clone()
            .delimited_by(just(Token::Ctrl('(')), just(Token::Ctrl(')')));

        let braces = expr
            .clone()
            .delimited_by(just(Token::Ctrl('{')), just(Token::Ctrl('}')));

        // Defines the parser for the primary expression. It is the
        // base of the expression parser.
        let primary = value
            .or(braces)
            .or(parenthesis)
            .or(brackets)
            .labelled("primary");

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
                    state.intern((expr, span))
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
                    state.intern((expr, span))
                },
            )
            .labelled("term");

        term
    });

    // Defines the parser for the equation. It is the base of the
    // parser of equations and inequations.
    expr_parser
        .clone()
        .then(
            just(Token::Identifier("="))
                .to(EqKind::Eq)
                .or(just(Token::Identifier("!=")).to(EqKind::Neq)),
        )
        .then(expr_parser.clone())
        .map(|((lhs, op), rhs)| Equation { kind: op, lhs, rhs })
        .map_with_span(|equation, span| (equation, span))
        .labelled("equation")
}

/// Parses a string into an [`Equation`].
///
/// [`Equation`]: [`Equation`]
fn parse(s: &str, state: &mut TermArena) -> Spanned<Equation> {
    type AriadneSpan = (String, std::ops::Range<usize>);

    // Defines the filename of the source. And it is used to
    // create the report.
    let filename = "terminal".to_string();

    let (tokens, lex_errors) = lexer().parse(s).into_output_errors();
    let tokens = tokens.unwrap_or_default();
    let tokens = tokens.as_slice().spanned((s.len()..s.len()).into());
    let (expr, errors) = parser()
        .parse_with_state(tokens, state)
        .into_output_errors();

    // If there are no errors, return the parsed expression.
    if !errors.is_empty() || !lex_errors.is_empty() {
        errors
            .into_iter()
            .map(|error| error.map_token(|c| c.to_string()))
            .chain(
                lex_errors
                    .into_iter()
                    .map(|error| error.map_token(|token| token.to_string())),
            )
            .for_each(|error| {
                Report::<AriadneSpan>::build(ReportKind::Error, filename.clone(), 0)
                    .with_code(1)
                    .with_message(error.to_string())
                    .with_label(
                        Label::new((filename.clone(), error.span().into_range()))
                            .with_message(error.reason().to_string())
                            .with_color(Color::Red),
                    )
                    .with_labels(error.contexts().map(|(label, span)| {
                        Label::new((filename.clone(), span.into_range()))
                            .with_message(format!("while parsing this {}", label))
                            .with_color(Color::Yellow)
                    }))
                    .finish()
                    .eprint((filename.to_string(), Source::from(s.to_string())))
                    .unwrap();
            });
    }

    // If the expression is not present, we return an error sentinel
    // value to avoid crashing.
    expr.unwrap_or_else(|| {
        let span: Span = (s.len()..s.len()).into();
        let equation = Equation {
            kind: EqKind::Eq,
            lhs: state.intern((Expr::Error, span)),
            rhs: state.intern((Expr::Error, span)),
        };
        (equation, span)
    })
}

/// The type of the error that can be returned by the unification
#[derive(Debug, Clone)]
pub enum TypeError {
    /// The two terms are not unifiable.
    NotUnifiable(Expr, Expr),

    /// The two terms are not compatible.
    IncompatibleOp(Op, Op),
}

/// Gets the precedence of an operator.
pub fn op_power(op: Op) -> usize {
    match op {
        Op::Add | Op::Sub => 1,
        Op::Mul | Op::Div => 2,
    }
}

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

    /// Rewrites a term to its normal form.
    pub fn rewrite(self, state: &mut TermArena) -> Term {
        self.apply_distributive_property(state)
            .apply_associativity(state)
            .whnf(state)
    }

    /// Reduces a term to its weak head normal form.
    pub fn whnf(self, state: &mut TermArena) -> Term {
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

    /// Unifies two terms. It's the main point of the equation.
    ///
    /// The logic relies here.
    pub fn unify(self, another: Term, state: &mut TermArena) -> Result<(), TypeError> {
        match (&state.get(self).0, &state.get(another).0) {
            // Errors are sentinel values, so they are threated like holes
            // and they are ignored, anything unifies with them.
            (Expr::Error, _) => {}
            (_, Expr::Error) => {}

            // If they are the same, they unify.
            (Expr::Number(_), Expr::Number(_)) => {}

            // If the term is a number, we try the following steps given the example:
            //   9 = x + 6
            //
            // 1. We get the reverse of `+`, which is `-`.
            // 2. We subtract `6` from both sides, to equate it,
            //    and since the `9` is a constant, we can reduce
            //    it to the WHNF.
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

            // If they are variables, they unify if they are the same.
            //
            // This check isn't inehenterly necessary, but it's a good catcher
            // to avoid panics, because if the variables are the same, they will
            // try to borrow the same data, and it will panic with ref cells.
            (Expr::Variable(variable_a), Expr::Variable(variable_b))
                if variable_a.name == variable_b.name => {}

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
        }

        Ok(())
    }
}

fn main() {
    let mut state = TermArena::default();
    let (equation, _) = parse("10 = x + 7", &mut state);
    print!("Input: {:?}", equation.lhs.debug(&state));
    print!(" = ");
    println!("{:?}", equation.rhs.debug(&state));

    let lhs = equation.lhs.rewrite(&mut state);
    let rhs = equation.rhs.rewrite(&mut state);
    lhs.unify(rhs, &mut state).unwrap();

    let resolutions = state
        .resolutions()
        .into_iter()
        .map(|(incognito, term)| format!("{incognito} = {:?}", term.debug(&state)))
        .collect::<Vec<_>>()
        .join(", ");

    println!("Output: {resolutions}");
}
