"""损失函数模块。"""

from siamese.losses.infonce import InfoNCELoss, OrientationAwareInfoNCELoss
from siamese.losses.gallery_ce import GalleryClassificationLoss

__all__ = ["InfoNCELoss", "OrientationAwareInfoNCELoss", "GalleryClassificationLoss"]