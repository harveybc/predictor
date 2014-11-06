<?php
$this->breadcrumbs = array(
	'Evaluaciones Generales',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Evaluaciones Generales', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Evaluaciones Generales', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Evaluaciones Generales'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
