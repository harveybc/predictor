<?php
$this->breadcrumbs = array(
	'Secuencias',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Secuencias', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Secuencias', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Secuencias'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
