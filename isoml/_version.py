"""
 isoml-1 (c) by Xin Han

 isoml-1 is licensed under a
 Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

 You should have received a copy of the license along with this
 work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""


TYPE_CHECKING = False
if TYPE_CHECKING:
    from typing import Tuple, Union
    VERSION_TUPLE = Tuple[Union[int, str], ...]
else:
    VERSION_TUPLE = object

version: str
__version__: str
__version_tuple__: VERSION_TUPLE
version_tuple: VERSION_TUPLE

__version__ = version = '0.1.dev18+gb807c8f.d20240619'
__version_tuple__ = version_tuple = (0, 1, 'dev18', 'gb807c8f.d20240619')
