<?php
$this->breadcrumbs = array(
	'Archivoses',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Create') . ' Archivos', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Manage') . ' Archivos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Archivoses'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
