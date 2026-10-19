/* small list helpers */
val is_cons : forall ('a : Type). list('a) -> bool
function is_cons(l) = match l { [||] => false, _ :: _ => true }

val reverse_list : forall ('a : Type). list('a) -> list('a)
function reverse_list(l) = {
  var r : list('a) = [||];
  var x : list('a) = l;
  while is_cons(x) do {
    match x { h :: t => { r = h :: r; x = t }, [||] => () }
  };
  r
}

/* l1 followed by l2 */
val pend_append : forall ('a : Type). (list('a), list('a)) -> list('a)
function pend_append(l1, l2) = {
  var r : list('a) = l2;
  var x : list('a) = reverse_list(l1);
  while is_cons(x) do {
    match x { h :: t => { r = h :: r; x = t }, [||] => () }
  };
  r
}

val list_length : forall ('a : Type). list('a) -> int
function list_length(l) = {
  var n : int = 0;
  var x : list('a) = l;
  while is_cons(x) do { match x { _ :: t => { n = n + 1; x = t }, [||] => () } };
  n
}

val sdrop : (string, int) -> string
function sdrop(s, n) = { let k = max_int(n, 0); assert(k >= 0); string_drop(s, k) }
val stake : (string, int) -> string
function stake(s, n) = { let k = max_int(n, 0); assert(k >= 0); string_take(s, k) }
