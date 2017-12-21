Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
        >>> df
           A  B
        0  1  2
        1  3  4
        >>> df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
        >>> df.append(df2)
           A  B
        0  1  2
        1  3  4
        0  5  6
        1  7  8
        With `ignore_index` set to True:
        >>> df.append(df2, ignore_index=True)
           A  B
        0  1  2
        1  3  4
        2  5  6
        3  7  8
