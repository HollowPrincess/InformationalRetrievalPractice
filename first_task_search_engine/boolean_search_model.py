def intersect_postings_lists(postings_left: dict, postings_right: dict) -> dict:
    """
    AND operation for two postings lists
    """
    left_items_iterator = iter(postings_left.items())  # type: iter
    right_items_iterator = iter(postings_right.items())  # type: iter
    ans = {}  # type: dict
    stopped_iterator = False  # type:bool

    # read first items from postings lists:
    try:
        left_key, left_value = next(left_items_iterator)  # types: str, int
    except StopIteration:
        stopped_iterator = True

    try:
        right_key, right_value = next(right_items_iterator)  # types: str, int
    except StopIteration:
        stopped_iterator = True

    while not stopped_iterator:
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
    AND operation for two and more postings lists
    """
    # pop the longest element from list of posting_lists
    postings_lists.sort(key=len)
    ans = postings_lists.pop()  # type: dict
    while ans and postings_lists:
        poped = postings_lists.pop()  # type: dict
        ans = intersect_postings_lists(ans, poped)
    return ans


def get_tail_for_not_stopped_iter(
    stopped_iterator: iter,
    left_items_iterator: iter,
    right_items_iterator: iter,
    ans: dict,
    left_key: str,
    left_value: int,
) -> dict:
    # this function is for OR operator
    # it was written because union function was too complex
    # if one iterator get end of the posting list
    # we need to add tail of other posting list
    if stopped_iterator == left_items_iterator:
        tail = dict(right_items_iterator)
    elif stopped_iterator == right_items_iterator:
        tail = dict(left_items_iterator)
    if not (left_key in ans):
        ans[left_key] = left_value

    if tail:
        ans.update(tail)
    return ans


def union_postings_lists(postings_left: dict, postings_right: dict) -> dict:
    """
    OR operation
    """
    left_items_iterator = iter(postings_left.items())
    right_items_iterator = iter(postings_right.items())
    stopped_iterator = False  # init value for running loop
    ans = {}  # type:dict

    # read first items from postings lists:
    try:
        left_key, left_value = next(left_items_iterator)  # types: str, int
    except StopIteration:
        stopped_iterator = True
        ans = postings_right

    try:
        right_key, right_value = next(right_items_iterator)  # types: str, int
    except StopIteration:
        stopped_iterator = True
        ans = postings_left

    while not stopped_iterator:
        if left_key == right_key:
            ans[left_key] = left_value + right_value  # it is a term frequency
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
    ans = get_tail_for_not_stopped_iter(
        stopped_iterator,
        left_items_iterator,
        right_items_iterator,
        ans,
        left_key,
        left_value,
    )
    return ans


def subtract_postings_lists(postings_left: dict, postings_right: dict) -> dict:
    """
    NOT operation
    """
    left_items_iterator = iter(postings_left.items())  # type:iter
    right_items_iterator = iter(postings_right.items())  # type:iter
    ans = {}  # type: dict
    stopped_iterator = False  # type:bool

    # read first items from postings lists:
    try:
        left_key, left_value = next(left_items_iterator)  # types: str, int
    except StopIteration:
        stopped_iterator = True

    try:
        right_key, right_value = next(right_items_iterator)  # types: str, int
    except StopIteration:
        stopped_iterator = True
        ans = postings_left

    check_tail_in_left_flag = False  # type:bool

    while not stopped_iterator:
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
    if (not (left_key in ans)) and check_tail_in_left_flag:
        ans[left_key] = left_value
        # we read item from left dict
        # and after that right dict was stopped
    if tail:
        ans.update(tail)
    return ans
