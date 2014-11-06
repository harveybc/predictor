<?php
$this->breadcrumbs = array(
	'Avisos Zis',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	//array('label'=>Yii::t('app', 'Crear') . ' Aviso ZI', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Avisos ZI', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Avisos Zi'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
