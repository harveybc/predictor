<?php
$this->breadcrumbs = array(
	'Tipo de Contenidos',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Tipo de Contenidos', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Tipo de Contenidos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Tipo de Contenidos'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
