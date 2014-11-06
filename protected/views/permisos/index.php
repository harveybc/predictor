<?php
$this->breadcrumbs = array(
	'Permisos',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Permisos', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Permisos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Permisos'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
