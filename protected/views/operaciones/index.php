<?php
$this->breadcrumbs = array(
	'Operaciones',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Operaciones', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Operaciones', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Operaciones'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
