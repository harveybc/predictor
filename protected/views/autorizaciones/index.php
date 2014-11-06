<?php
$this->breadcrumbs = array(
	'Autorizaciones',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Nueva') . ' Autorización', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Autorizaciones', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Autorizaciones'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
