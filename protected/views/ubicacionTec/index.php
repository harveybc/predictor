<?php
$this->breadcrumbs = array(
	'Ubicación Técnica',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Ubicación Técnica', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Ubicación Técnica', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Ubicación Técnica'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
