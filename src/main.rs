use std::{cell::RefCell, fmt::Display, rc::Rc};

use ariadne::{Color, Label, Report, ReportKind, Source};
use chumsky::{
    extra::{self, Err},
    prelude::{Input, Rich},
    primitive::{just, one_of},
    recovery::{nested_delimiters, via_parser},
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
#[derive(Debug, Clone, Copy)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

/// A binary operation. This is a binary operation that takes two arguments.
#[derive(Debug, Clone)]
pub struct BinOp {
    pub op: Op,
    pub lhs: Box<Spanned<Term>>,
    pub rhs: Box<Spanned<Term>>,
}

/// A function. This is a function that takes arguments.
#[derive(Debug, Clone)]
pub enum Function {
    Pow {
        base: Box<Spanned<Term>>,
        exp: Box<Spanned<Term>>,
    },
    Sqrt(Box<Spanned<Term>>),
    Cbrt(Box<Spanned<Term>>),
    Factorial(Box<Spanned<Term>>),
}

/// A variable. This is a variable that can be assigned to.
#[derive(Default, Debug, Clone)]
pub struct Variable {
    pub data: Rc<RefCell<Term>>,
}

/// A term in the calculator. This is the lowest level of the calculator.
#[derive(Default, Debug, Clone)]
pub enum Term {
    #[default]
    Error,
    Number(usize),
    Decimal(usize, usize),
    Group(Box<Spanned<Term>>),
    Variable(String, Variable),
    BinOp(BinOp),
    Apply(Function),
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
    pub lhs: Spanned<Term>,
    pub rhs: Spanned<Term>,
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
    extra::Err<Rich<'tokens, Token<'src>, Span>>,
> {
    // Defines the parser for the expression. It is recursive, because
    // it can be nested.
    let expr_parser = recursive(|expr| {
        // Defines the parser for the value. It is the base of the
        // expression parser.
        let value = select! {
            Token::Number(number) => Term::Number(number),
            Token::Decimal(int, decimal) => Term::Decimal(int, decimal),
            Token::Identifier(identifier) => Term::Variable(identifier.into(), Variable::default()),
        }
        .labelled("value");

        // Defines the parser for the primary expression. It is the
        // base of the expression parser.
        let primary = value
            .or(expr
                .clone()
                .delimited_by(just(Token::Ctrl('(')), just(Token::Ctrl(')')))
                .map(|expr| Term::Group(Box::new(expr))))
            .or(expr
                .clone()
                .delimited_by(just(Token::Ctrl('[')), just(Token::Ctrl(']')))
                .boxed()
                .map(|expr| Term::Group(Box::new(expr))))
            .or(expr
                .clone()
                .delimited_by(just(Token::Ctrl('{')), just(Token::Ctrl('}')))
                .boxed()
                .map(|expr| Term::Group(Box::new(expr))))
            .map_with_span(|expr, span| (expr, span))
            .recover_with(via_parser(nested_delimiters(
                Token::Ctrl('('),
                Token::Ctrl(')'),
                [
                    (Token::Ctrl('['), Token::Ctrl(']')),
                    (Token::Ctrl('{'), Token::Ctrl('}')),
                ],
                |span| (Term::Error, span),
            )))
            .labelled("primary");

        let factor = primary
            .clone()
            .then(just(Token::Identifier("!")).or_not())
            .map(|(expr, not)| match not {
                Some(_) => Term::Apply(Function::Factorial(expr.into())),
                None => expr.0,
            })
            .map_with_span(|expr, span| (expr, span))
            .labelled("factor");

        let add = factor
            .clone()
            .foldl(
                just(Token::Identifier("+"))
                    .to(Op::Add)
                    .or(just(Token::Identifier("-")).to(Op::Sub))
                    .then(expr.clone())
                    .repeated(),
                |lhs: Spanned<Term>, (op, rhs)| {
                    let span = SimpleSpan::new(lhs.1.start, rhs.1.end);
                    let expr = Term::BinOp(BinOp {
                        op,
                        lhs: lhs.into(),
                        rhs: rhs.into(),
                    });
                    (expr, span)
                },
            )
            .labelled("add");

        let mul = add
            .clone()
            .foldl(
                just(Token::Identifier("*"))
                    .to(Op::Mul)
                    .or(just(Token::Identifier("/")).to(Op::Div))
                    .then(expr.clone())
                    .repeated(),
                |lhs: Spanned<Term>, (op, rhs)| {
                    let span = SimpleSpan::new(lhs.1.start, rhs.1.end);
                    let expr = Term::BinOp(BinOp {
                        op,
                        lhs: lhs.into(),
                        rhs: rhs.into(),
                    });
                    (expr, span)
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
fn parse(s: &str) -> Spanned<Equation> {
    type AriadneSpan = (String, std::ops::Range<usize>);

    // Defines the filename of the source. And it is used to
    // create the report.
    let filename = "terminal".to_string();

    let (tokens, lex_errors) = lexer().parse(s).into_output_errors();
    let tokens = tokens.unwrap_or_default();
    let (expr, errors) = parser()
        .parse(tokens.as_slice().spanned((s.len()..s.len()).into()))
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
            lhs: (Term::Error, span),
            rhs: (Term::Error, span),
        };
        (equation, span)
    })
}

/// Unifies two terms. It's the main point of the equation.
///
/// The logic relies here.
pub fn unify(term: Term, another: Term) {}

fn main() {
    println!("{:?}", parse("1 + 2 * 3 = x"));
}
