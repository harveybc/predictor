<?php
$this->breadcrumbs = array(
	'Módulos',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Módulos', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Módulos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Módulos'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
