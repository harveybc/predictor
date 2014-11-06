<?php
$this->breadcrumbs = array(
	'Evaluaciones',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Evaluaciones', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Evaluaciones', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Evaluaciones'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
