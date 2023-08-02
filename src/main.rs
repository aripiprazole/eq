use std::fmt::Display;

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
    pub lhs: Box<Spanned<Expr>>,
    pub rhs: Box<Spanned<Expr>>,
}

/// A function. This is a function that takes arguments.
#[derive(Debug, Clone)]
pub enum Function {
    Pow {
        base: Box<Spanned<Expr>>,
        exp: Box<Spanned<Expr>>,
    },
    Sqrt(Box<Spanned<Expr>>),
    Cbrt(Box<Spanned<Expr>>),
    Factorial(Box<Spanned<Expr>>),
}

/// A term in the calculator. This is the lowest level of the calculator.
#[derive(Debug, Clone)]
pub enum Expr {
    Error,
    Number(usize),
    Decimal(usize, usize),
    Group(Box<Spanned<Expr>>),
    Identifier(String),
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
    pub lhs: Spanned<Expr>,
    pub rhs: Spanned<Expr>,
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
        });

    let op = one_of("+*-/!^|&<>=")
        .repeated()
        .at_least(1)
        .map_slice(Token::Identifier);

    let ctrl = one_of("()[]{}").map(Token::Ctrl);

    num.or(op)
        .or(ctrl)
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
            Token::Number(number) => Expr::Number(number),
            Token::Decimal(int, decimal) => Expr::Decimal(int, decimal),
            Token::Identifier(identifier) => Expr::Identifier(identifier.into()),
        }
        .labelled("value");

        // Defines the parser for the primary expression. It is the
        // base of the expression parser.
        let primary = value
            .or(expr
                .clone()
                .delimited_by(just(Token::Ctrl('(')), just(Token::Ctrl(')')))
                .map(|expr| Expr::Group(Box::new(expr))))
            .or(expr
                .clone()
                .delimited_by(just(Token::Ctrl('[')), just(Token::Ctrl(']')))
                .boxed()
                .map(|expr| Expr::Group(Box::new(expr))))
            .or(expr
                .clone()
                .delimited_by(just(Token::Ctrl('{')), just(Token::Ctrl('}')))
                .boxed()
                .map(|expr| Expr::Group(Box::new(expr))))
            .map_with_span(|expr, span| (expr, span))
            .recover_with(via_parser(nested_delimiters(
                Token::Ctrl('('),
                Token::Ctrl(')'),
                [
                    (Token::Ctrl('['), Token::Ctrl(']')),
                    (Token::Ctrl('{'), Token::Ctrl('}')),
                ],
                |span| (Expr::Error, span),
            )))
            .labelled("primary");

        let factor = primary
            .clone()
            .then(just(Token::Identifier("!")).or_not())
            .map(|(expr, not)| match not {
                Some(_) => Expr::Apply(Function::Factorial(expr.into())),
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
                |lhs: Spanned<Expr>, (op, rhs)| {
                    let span = SimpleSpan::new(lhs.1.start, rhs.1.end);
                    let expr = Expr::BinOp(BinOp {
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
                |lhs: Spanned<Expr>, (op, rhs)| {
                    let span = SimpleSpan::new(lhs.1.start, rhs.1.end);
                    let expr = Expr::BinOp(BinOp {
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
    let filename = "test".to_string();
    let (tokens, lex_errors) = lexer().parse(s).into_output_errors();
    let tokens = tokens.unwrap();
    let (expr, errors) = parser()
        .parse(tokens.as_slice().spanned((s.len()..s.len()).into()))
        .into_output_errors();

    if !errors.is_empty() || !lex_errors.is_empty() {
        type AriadneSpan = (String, std::ops::Range<usize>);

        Report::<AriadneSpan>::build(ReportKind::Error, filename.clone(), 0)
            .with_code(1)
            .with_message("parse error")
            .with_labels(
                errors
                    .into_iter()
                    .map(|error| error.map_token(|c| c.to_string()))
                    .chain(
                        lex_errors
                            .into_iter()
                            .map(|error| error.map_token(|token| token.to_string())),
                    )
                    .map(|error| {
                        Label::new((filename.clone(), error.span().into_range()))
                            .with_message(error.reason().to_string())
                            .with_color(Color::Red)
                    }),
            )
            .finish()
            .eprint((filename.to_string(), Source::from(s.to_string())))
            .unwrap();
    }

    expr.unwrap()
}

fn main() {
    println!("{:?}", parse("(1 + 2) * 3 = 7"));
}
