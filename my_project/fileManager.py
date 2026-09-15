import pandas as pd

class FileManager():
    """
    A simple file manager.
        
    Parameters
    ----------
    format_ : str, optional
        Initial format of the files its managing
    
    Attributes
    ----------
    format : str | None
        Format of the files its managing
        
    Examples
    --------
    >>> fm = FileManager()
    >>> fm.set_format("parquet")
    >>> df = fm.read("parquet_data")
    
    >>> fm = FileManager("csv")
    >>> df = fm.read("csv_data")
    """
        
    formats = {
        'csv': {'write': lambda df, f: df.to_csv(f, index=False), 'read': pd.read_csv, 'ext': '.csv'},
        'parquet': {'write': lambda df, f: df.to_parquet(f), 'read': pd.read_parquet, 'ext': '.parquet'},
        'feather': {'write': lambda df, f: df.to_feather(f), 'read': pd.read_feather, 'ext': '.feather'},
        'pickle': {'write': lambda df, f: df.to_pickle(f), 'read': pd.read_pickle, 'ext': '.pkl'}
    }

    def __init__(self,*, format_: str | None = None):
        self.format = None
        if format_ is not None:
            self.set_format(format_)
        
    @staticmethod
    def avalible_formats() -> list[str]:
        """
        Static method. Obtains the supported file formats.
        
        Returns
        -------
        list[str]
            List of strings, each a supported format
        
        Examples
        --------
        >>> formats = FileManager.avalible_formats()
        """
        
        return list(FileManager.formats.keys())
        
    @staticmethod
    def is_format(format_: str) -> bool:
        """
        Static method. Checks if a file format is supported.
        
        Parameters
        ----------
        format_ : str
            File format to check
        
        Returns
        -------
        bool
            True if it is supported
        
        Examples
        --------
        >>> csv_supported = FileManager.is_format("csv")
        """
        
        return (format_ in FileManager.formats.keys())

    def set_format(self, format_: str) -> None:
        """
        Sets the file format of the files to manage.
        
        Parameters
        ----------
        format_ : str
            File format to use
        
        Examples
        --------
        >>> fm.set_format("csv")

        Raises
        ------
        Exception
            If the format is not supported
        """
        
        if not self.is_format(format_):
            raise f"{format_} format not supported. Avalible formats: {self.avalible_formats()}"
        self.format = format_

    def _get_name(self, input_filename: str) -> str:
        """
        Adds the apropiate extension to the filename.
        
        Parameters
        ----------
        input_filename : str
            Filename without extension
        
        Returns
        -------
        str
            Filename with the extension of the current format
        
        Examples
        --------
        >>> fm.set_format("csv")
        >>> csv_filename = fm._get_name("name")
        """
        
        filename = f'{input_filename}{self.formats[self.format]["ext"]}'
        return filename
        
    def read(self, input_filename: str) -> pd.DataFrame:
        """
        Reads from the selected file.
        
        Parameters
        ----------
        input_filename : str
            Filename without extension
        
        Returns
        -------
        pd.DataFrame
            Dataframe with the content read from the selected file
        
        Examples
        --------
        >>> fm.set_format("csv")
        >>> df = fm.read("csv_file")
        """
        
        filename = self._get_name(input_filename)
        return self.formats[self.format]["read"](filename)
        
    def write(self, df: pd.DataFrame, input_filename: str) -> str | None:
        """
        Writes to the selected file.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containg the content to write
        input_filename : str
            Filename without extension
        
        Returns
        -------
        str | None
            Passes the return of the apropiate write call
        
        Examples
        --------
        >>> fm.set_format("csv")
        >>> fm.write(df, "csv_file")
        """
        
        filename = self._get_name(input_filename)
        return self.formats[self.format]["write"](df, filename)