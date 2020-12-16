\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{thesis}[2020/11/16 Styling for master thesis]

\newif\if@print
\DeclareOption{print}{\@printtrue}
\DeclareOption{digital}{\@printfalse}

\ProcessOptions\relax

% General
\LoadClassWithOptions{scrreprt}

% Language
\usepackage{polyglossia}
\setmainlanguage{english}
\setotherlanguage[spelling=old,babelshorthands=true]{german}

% Geometry
\usepackage[onehalfspacing]{setspace}
\usepackage[a4paper,width=150mm,top=30mm,bottom=30mm, bindingoffset=1cm]{geometry}

% Links
\usepackage{color,xcolor}
\usepackage{hyperref}
\if@print % switch link coloring off for printing
	\hypersetup{colorlinks=false}
\else
	\hypersetup{colorlinks=true}
\fi
\definecolor{uhd}{RGB}{196,19,47} % the red UHD brand color
\hypersetup{
	allcolors=uhd
}
\AtBeginDocument{
	\hypersetup{ % set pdf metadata
	    pdfauthor={\@author},
		pdftitle={\@title}
	}
}

% Fonts
\usepackage{fontspec, xunicode}
% \setmainfont{EMERITA Latina}
\setsansfont{EB Garamond}
% \setmonofont[Scale=MatchLowercase]{Menlo}

% Units
\usepackage[separate-uncertainty]{siunitx}
\newcommand{\around}{{\sim}}

% Bibliography
% authoryear
\usepackage[style=apa,backend=biber,maxcitenames=1,uniquename=false,uniquelist=minyear,url=false]{biblatex}

%\nocite{*} % print all references, even those not used

% Page Layout
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\if@print
\fancyhead[LE]{
	\textit{\nouppercase{\leftmark}}
}
\fancyhead[RO]{
	\textit{\nouppercase{\rightmark}}
}
\fancyfoot[RO,LE]{
	\thepage
}
\else
\fancyhead[L]{
	\textit{\nouppercase{\leftmark}}
}
\fancyfoot[C]{
	\thepage
}
\fi
\renewcommand{\headrulewidth}{0.2pt}
\renewcommand{\footrulewidth}{0pt}
\fancypagestyle{plain}{%
	\fancyhf{}
	\if@print
	\fancyfoot[RO,LE]{
		\thepage
	}
	\else
	\fancyfoot[C]{
		\thepage
	}
	\fi
	\renewcommand{\headrulewidth}{0pt}
	\renewcommand{\footrulewidth}{0pt}
}

% YOLO
\renewcommand{\chaptername}{}
% \renewcommand{\chaptermark}[1]{\markboth{#1}{}}
% \renewcommand{\sectionmark}[1]{\markright{\thesubsection\ #1}}

% Appendix
\usepackage[toc,page]{appendix}

% Remove all end-of-counter dots
\renewcommand{\autodot}{}

\endinput