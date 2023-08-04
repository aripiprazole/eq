use std::{cell::RefCell, collections::HashMap, fmt::Display, hash::Hash, ops::Range, rc::Rc};

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
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct BinOp {
    pub op: Op,
    pub lhs: Term,
    pub rhs: Term,
}

/// A function. This is a function that takes arguments.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Function {
    Pow { base: Term, exp: Term },
    Sqrt(Term),
    Cbrt(Term),
    Factorial(Term),
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
        self.data.borrow().clone()
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
pub enum TermKind {
    #[default]
    Error,
    Number(usize),
    Decimal(usize, usize),
    Group(Term),
    Variable(Variable),
    BinOp(BinOp),
    Apply(Function),
}

/// Shows a term in a human-readable format.
pub fn show(term: Term, fuel: usize, state: &mut TermArena) -> String {
    if fuel == 0 {
        return "…".into();
    }

    match &state.get(term).0 {
        TermKind::Error => "error".into(),
        TermKind::Number(n) => format!("{n}"),
        TermKind::Decimal(n, decimal) => format!("{n}.{decimal}"),
        TermKind::Group(group) => format!("({})", show(*group, fuel - 1, state)),
        TermKind::Variable(variable) => match variable.data() {
            Some(value) => format!("{}", show(value, fuel - 1, state)),
            None => format!("?{}", variable.name),
        },
        TermKind::BinOp(bin_op) => {
            let lhs = show(bin_op.lhs, fuel - 1, state);
            let rhs = show(bin_op.rhs, fuel - 1, state);
            let op_str = match bin_op.op {
                Op::Add => "+",
                Op::Sub => "-",
                Op::Mul => "*",
                Op::Div => "/",
            };

            format!("{lhs} {op_str} {rhs}")
        }
        TermKind::Apply(_) => todo!(),
    }
}

#[derive(Default, Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Term(usize);

#[derive(Default)]
pub struct TermArena {
    pub id_to_slot: HashMap<Term, Rc<Spanned<TermKind>>>,
    pub slot_to_id: HashMap<TermKind, Term>,
}

impl TermArena {
    /// Creates a new term in the arena
    pub fn insert(&mut self, term: Spanned<TermKind>) -> Term {
        let id = Term(fxhash::hash(&term.0));
        self.id_to_slot.insert(id, Rc::new(term.clone()));
        self.slot_to_id.insert(term.0, id);
        id
    }

    /// Gets the term from the arena.
    pub fn get(&self, id: Term) -> Rc<Spanned<TermKind>> {
        self.id_to_slot
            .get(&id)
            .cloned()
            .unwrap_or_else(|| Rc::new((TermKind::Error, Range::<usize>::default().into())))
    }

    /// Gets the term from the arena.
    pub fn resolutions(&self) -> Vec<(String, Term)> {
        self.id_to_slot
            .iter()
            .filter_map(|(_, slot)| {
                if let TermKind::Variable(variable) = &slot.0 {
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
                Token::Number(number) => TermKind::Number(number),
                Token::Decimal(int, decimal) => TermKind::Decimal(int, decimal),
                Token::Identifier(identifier) => TermKind::Variable(Variable { name: identifier.into(), data: Rc::default() }),
            }
            .map_with_span(|kind, span| (kind, span))
            .map_with_state(|term, _, state: &mut TermArena| state.insert(term))
            .labelled("value");

        let brackets = expr
            .clone()
            .delimited_by(just(Token::Ctrl('[')), just(Token::Ctrl(']')))
            .map(TermKind::Group)
            .map_with_span(|expr, span| (expr, span))
            .map_with_state(|term, _, state: &mut TermArena| state.insert(term));

        let parenthesis = expr
            .clone()
            .delimited_by(just(Token::Ctrl('(')), just(Token::Ctrl(')')))
            .map(TermKind::Group)
            .map_with_span(|expr, span| (expr, span))
            .map_with_state(|term, _, state: &mut TermArena| state.insert(term));

        let braces = expr
            .clone()
            .delimited_by(just(Token::Ctrl('{')), just(Token::Ctrl('}')))
            .map(TermKind::Group)
            .map_with_span(|expr, span| (expr, span))
            .map_with_state(|term, _, state: &mut TermArena| state.insert(term));

        // Defines the parser for the primary expression. It is the
        // base of the expression parser.
        let primary = value
            .or(braces)
            .or(parenthesis)
            .or(brackets)
            .labelled("primary");

        let factor = primary
            .clone()
            .then(just(Token::Identifier("!")).or_not())
            .map_with_state(|(expr, not), span, state: &mut TermArena| match not {
                Some(_) => {
                    let function = Function::Factorial(expr);
                    let kind = TermKind::Apply(function);

                    state.insert((kind, span))
                }
                None => expr,
            })
            .labelled("factor");

        let add = factor
            .clone()
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
                    let expr = TermKind::BinOp(BinOp { op, lhs, rhs });
                    state.insert((expr, span))
                },
            )
            .labelled("add");

        let mul = add
            .clone()
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
                    let expr = TermKind::BinOp(BinOp { op, lhs, rhs });
                    state.insert((expr, span))
                },
            )
            .labelled("mul");

        mul
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
            lhs: state.insert((TermKind::Error, span)),
            rhs: state.insert((TermKind::Error, span)),
        };
        (equation, span)
    })
}

/// The type of the error that can be returned by the unification
#[derive(Debug, Clone)]
pub enum TypeError {
    /// The two terms are not unifiable.
    NotUnifiable(TermKind, TermKind),

    /// The two terms are not compatible.
    IncompatibleOp(Op, Op),
}

/// Reduces a term to its weak head normal form.
pub fn whnf(term: Term, state: &mut TermArena) -> Term {
    let (kind, span) = &*state.get(term);
    let new_kind = match kind {
        TermKind::Group(group) => TermKind::Group(whnf(*group, state)),
        TermKind::BinOp(bin_op) => {
            let lhs = whnf(bin_op.lhs, state);
            let rhs = whnf(bin_op.rhs, state);

            // If the term is a number, we try to reduce it evaluating
            // the operation.
            match &state.get(lhs).0 {
                // If the term is a number
                // 
                // We assume that the term is a number and we try to
                // reduce it.
                TermKind::Number(lhs) => match &state.get(rhs).0 {
                    TermKind::Number(rhs) => {
                        let number = match bin_op.op {
                            Op::Add => lhs + rhs,
                            Op::Sub => lhs - rhs,
                            Op::Mul => lhs * rhs,
                            Op::Div => lhs / rhs,
                        };

                        TermKind::Number(number)
                    }
                    TermKind::Decimal(_number, _decimal) => return term,
                    _ => return term,
                },

                // If the term is a decimal
                // 
                // We assume that the term is a decimal and we try to
                // reduce it.
                TermKind::Decimal(_number, _decimal) => match &state.get(rhs).0 {
                    TermKind::Number(_rhs) => return term,
                    TermKind::Decimal(_number, _decimal) => return term,
                    _ => return term,
                },
                _ => return term,
            }
        }
        // If the term is a variable, we try to reduce it.
        TermKind::Variable(hole) => match hole.data() {
            Some(value) => return whnf(value, state),
            None => kind.clone(),
        },
        _ => kind.clone(),
    };
    state.insert((new_kind, *span))
}

/// Unifies two terms. It's the main point of the equation.
///
/// The logic relies here.
pub fn unify(term: Term, another: Term, state: &mut TermArena) -> Result<(), TypeError> {
    match (&state.get(term).0, &state.get(another).0) {
        // Errors are sentinel values, so they are threated like holes
        // and they are ignored, anything unifies with them.
        (TermKind::Error, _) => {}
        (_, TermKind::Error) => {}

        // If they are the same, they unify.
        (TermKind::Number(_), TermKind::Number(_)) => {}
        (TermKind::Decimal(_, _), TermKind::Decimal(_, _)) => {}

        // If they are variables, they unify if they are the same.
        //
        // This check isn't inehenterly necessary, but it's a good catcher
        // to avoid panics, because if the variables are the same, they will
        // try to borrow the same data, and it will panic with ref cells.
        (TermKind::Variable(variable_a), TermKind::Variable(variable_b))
            if variable_a.name == variable_b.name => {}

        // Unifies the variable with the term, if the variable is not bound. If
        // it's bound, it will try to unify, if it's not unifiable, it will
        // return an error.
        (_, TermKind::Variable(variable)) => {
            match variable.data() {
                // If the variable is already bound, we unify the bound
                Some(bound) => {
                    unify(term, bound, state)?;
                }
                // Empty hole
                None => {
                    variable.data.replace(Some(term));
                }
            }
        }
        (TermKind::Variable(variable), _) => {
            match variable.data() {
                // If the variable is already bound, we unify the bound
                Some(bound) => {
                    unify(term, bound, state)?;
                }
                // Empty hole
                None => {
                    variable.data.replace(Some(another));
                }
            }
        }

        // Unifies the bin ops if they are the same, and unifies the
        // operands.
        (TermKind::BinOp(bin_op_a), TermKind::BinOp(bin_op_b)) => {
            if bin_op_a.op == bin_op_b.op {
                let lhs_a = whnf(bin_op_a.lhs, state);
                let rhs_a = whnf(bin_op_a.rhs, state);

                let lhs_b = whnf(bin_op_b.lhs, state);
                let rhs_b = whnf(bin_op_b.rhs, state);

                unify(lhs_a, lhs_b, state)?;
                unify(rhs_a, rhs_b, state)?;
            } else {
                return Err(TypeError::IncompatibleOp(bin_op_a.op, bin_op_b.op));
            }
        }

        // Reduce groups to the normal form
        (_, TermKind::Group(another)) => {
            let term = whnf(term, state);
            let another = whnf(*another, state);

            unify(term, another, state)?;
        }
        (TermKind::Group(term), _) => {
            let term = whnf(*term, state);
            let another = whnf(another, state);

            unify(term, another, state)?;
        }

        // If they aren't compatible, return an error.
        (kind_a, kind_b) => return Err(TypeError::NotUnifiable(kind_a.clone(), kind_b.clone())),
    }

    Ok(())
}

fn main() {
    let mut state = TermArena::default();
    let (equation, _) = parse("3 * 3 = x + 6", &mut state);
    let lhs = whnf(equation.lhs, &mut state);
    let rhs = whnf(equation.rhs, &mut state);
    unify(lhs, rhs, &mut state).unwrap();

    let resolutions = state
        .resolutions()
        .into_iter()
        .map(|(incognito, term)| {
            let term = show(term, 256, &mut state);

            format!("{incognito} = {term}")
        })
        .collect::<Vec<_>>()
        .join(", ");

    println!("{resolutions}");
}
