%childOf(X,Y) - X is a child of Y
%Facts for both parents
childOf(andrew,elizabeth).
childOf(andrew,philip).
childOf(anne,elizabeth).
childOf(anne,philip).
childOf(beatrice,andrew).
childOf(beatrice,sarah).
childOf(charles,elizabeth).
childOf(charles,philip).
childOf(diana,kydd).
childOf(diana,spencer).
childOf(edward,elizabeth).
childOf(edward,philip).
childOf(elizabeth,george).
childOf(elizabeth,mum).
childOf(eugenie,andrew).
childOf(eugenie,sarah).
childOf(harry,charles).
childOf(harry,diana).
childOf(james,edward).
childOf(james,sophie).
childOf(louise,edward).
childOf(louise,sophie).
childOf(margaret,george).
childOf(margaret,mum).
childOf(peter,anne).
childOf(peter,mark).
childOf(william,charles).
childOf(william,diana).
childOf(zara,anne).
childOf(zara,mark).

female(anne).
female(beatrice).
female(diana).
female(elizabeth).
female(kydd).
female(louise).
female(margaret).
female(mum).
female(sarah).
female(sophie).
female(zara).

male(andrew).
male(charles).
male(edward).
male(eugenie).
male(george).
male(harry).
male(james).
male(mark).
male(peter).
male(philip).
male(spencer).
male(william).

%base cases for spouse
married(anne,mark).
married(diana,charles).
married(elizabeth,philip).
married(kydd,spencer).
married(mum,george).
married(sarah,andrew).
married(sophie,edward).


% PROLOG PROGRAM %

spouse(X,Y) :- married(X,Y);married(Y,X).

sonOf(S,P) :- male(S),childOf(S,P).
daughterOf(D,P) :- female(D),childOf(D,P).

brotherOf(X,Y) :- male(X),childOf(X,P),childOf(Y,P),X\=Y.
sisterOf(X,Y) :- female(X),childOf(X,P),childOf(Y,P),X\=Y.
siblingOf(X,Y) :- brotherOf(X,Y);
    			  sisterOf(X,Y).

grandchildOf(X,GP) :- childOf(X,P),childOf(P,GP).

ancestorOf(X,Y) :- childOf(Y,X);
    			   childOf(Y,Z),ancestorOf(X,Z).

uncleOf(X,Y) :- male(X),childOf(Y,Z),childOf(Z,W),childOf(X,W),X\=Z.
auntOf(X,Y) :- female(X),childOf(Y,Z),childOf(Z,W),childOf(X,W),X\=Z.

firstCousinOf(X,Y) :- childOf(X,W),childOf(Y,Z),siblingOf(W,Z).

brotherInLawOf(X,Y) :- spouse(Y,S),brotherOf(X,S);
    				 male(X),spouse(X,S),siblingOf(S,Y).
sisterInLawOf(X,Y) :- spouse(Y,S),sisterOf(X,S);
    				female(X),spouse(X,S),siblingOf(S,Y).

grandparentOf(C,GP) :- childOf(C,P),childOf(P,GP).