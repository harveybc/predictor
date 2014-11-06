<?php
$this->breadcrumbs = array(
	'Competencias',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Competencias', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Competencias', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Competencias'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
