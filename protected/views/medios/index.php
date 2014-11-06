<?php
$this->breadcrumbs = array(
	'Medios de Publicación',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Medios Publicación', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Medios Publicación', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Medios de Publicación'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
