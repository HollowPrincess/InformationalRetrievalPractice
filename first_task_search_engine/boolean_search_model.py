def intersect_postings_lists(postings_left: dict, postings_right: dict) -> dict:
    """
    AND operation for two postings lists
    """
    left_items_iterator = iter(postings_left.items())  # type: iter
    right_items_iterator = iter(postings_right.items())  # type: iter
    left_key, left_value = next(left_items_iterator)  # types: str, int
    right_key, right_value = next(right_items_iterator)  # types: str, int
    ans = {}  # type: dict

    while True:
        if left_key == right_key:
            ans[left_key] = left_value + right_value
            try:
                left_key, left_value = next(left_items_iterator)
                right_key, right_value = next(right_items_iterator)
            except StopIteration:
                # get end of one of postings lists, intersection is full
                break
        elif left_key < right_key:
            try:
                left_key, left_value = next(left_items_iterator)
            except StopIteration:
                break
        else:
            try:
                right_key, right_value = next(right_items_iterator)
            except StopIteration:
                break
    return ans


def intersect_many_postings_lists(postings_lists: list) -> dict:
    """
    AND operation for more than two postings lists
    """
    postings_lists.sort(key=len)
    ans = (
        postings_lists.pop()
    )  # type: dict #pop the longest element from list of posting_lists
    while (len(ans) > 0) and (len(postings_lists) > 0):
        poped = postings_lists.pop()  # type: dict
        ans = intersect_postings_lists(ans, poped)
    return ans


def union_postings_lists(postings_left: dict, postings_right: dict) -> dict:
    """
    OR operation
    """
    left_items_iterator = iter(postings_left.items())  # type: iter
    right_items_iterator = iter(postings_right.items())  # type: iter
    stopped_iterator = False  # init value for running loop
    left_key, left_value = next(left_items_iterator)  # types: str, int
    right_key, right_value = next(right_items_iterator)  # types: str, int
    ans = {}  # type dict

    while not stopped_iterator:
        if left_key == right_key:
            ans[left_key] = left_value + right_value
            try:
                left_key, left_value = next(left_items_iterator)
            except StopIteration:
                stopped_iterator = left_items_iterator
                break

            try:
                right_key, right_value = next(right_items_iterator)
            except StopIteration:
                stopped_iterator = right_items_iterator
                break

        elif left_key < right_key:
            ans[left_key] = left_value
            try:
                left_key, left_value = next(left_items_iterator)
            except StopIteration:
                stopped_iterator = left_items_iterator
                break
        else:
            ans[right_key] = right_value
            try:
                right_key, right_value = next(right_items_iterator)
            except StopIteration:
                stopped_iterator = right_items_iterator
                break

    # get tail of not stopped iterator
    if stopped_iterator == left_items_iterator:
        tail = dict(right_items_iterator)
    elif stopped_iterator == right_items_iterator:
        if not (left_key in ans.keys()):
            ans[left_key] = left_value
        tail = dict(left_items_iterator)

    if tail:
        ans.update(tail)
    return ans


def subtract_postings_lists(postings_left: dict, postings_right: dict) -> dict:
    """
    NOT operation
    """
    left_items_iterator = iter(postings_left.items())  # type:iter
    right_items_iterator = iter(postings_right.items())  # type:iter
    left_key, left_value = next(left_items_iterator)  # types: str, int
    right_key, right_value = next(right_items_iterator)  # types: str, int
    ans = {}  # type: dict
    check_tail_in_left_flag = False  # type:bool

    while True:
        if left_key == right_key:
            try:
                left_key, left_value = next(left_items_iterator)
            except StopIteration:
                break

            try:
                right_key, right_value = next(right_items_iterator)
            except StopIteration:
                check_tail_in_left_flag = True
                break
        elif left_key < right_key:
            ans[left_key] = left_value
            try:
                left_key, left_value = next(left_items_iterator)
            except StopIteration:
                break
        else:
            try:
                right_key, right_value = next(right_items_iterator)
            except StopIteration:
                check_tail_in_left_flag = True
                break
    # get tail of left postings if right postings ended
    tail = dict(left_items_iterator)
    if (not (left_key in ans.keys())) and check_tail_in_left_flag:
        ans[left_key] = left_value
    if tail:
        ans.update(tail)
    return ans
